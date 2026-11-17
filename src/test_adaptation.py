import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "safeshift_src"))
import torch
from torch import nn
import torch.nn.functional as F

from models import create_model, list_models
from adaptation import (SourceOnly, BatchNormAdapt, Tent, CoTTA, ETA, EATA, SAR,
                        create_adapter, estimate_fisher_from_unlabeled)


def tiny_bn():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8),
                         nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(8, 100))


def tiny_gn():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GroupNorm(4, 8),
                         nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(8, 100))


def test_models():
    for name in ('resnet20_cifar', 'resnet26_cifar', 'resnet18_cifar',
                 'preact_resnet18_cifar', 'wrn40_2', 'resnext29_augmix_arch',
                 'densenet_bc100_12', 'vgg16_bn_cifar'):
        model = create_model(name, num_classes=10).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10), name
        assert torch.isfinite(out).all()


def test_source_and_batchnorm():
    torch.manual_seed(0)
    backbone = tiny_bn().eval()
    original = copy.deepcopy(backbone.state_dict())
    x = torch.randn(4, 3, 16, 16)
    src = SourceOnly(backbone)
    with torch.no_grad():
        assert src(x).shape == (4, 100)
    bn = BatchNormAdapt(backbone)
    with torch.no_grad():
        assert bn(x).shape == (4, 100)
    assert all(torch.equal(a, backbone.state_dict()[k]) for k,a in original.items())
    bn.reset()
    assert torch.allclose(bn(x), bn(x))


def test_tent_reset_state_and_no_grad():
    torch.manual_seed(0)
    backbone = tiny_bn()
    adapter = Tent(backbone)
    conv_before = adapter.model[0].weight.detach().clone()
    bn_before = adapter.model[1].weight.detach().clone()
    with torch.no_grad():
        z = adapter(torch.randn(8, 3, 16, 16))
    assert z.shape == (8, 100) and adapter.n_updates == 1
    assert torch.equal(adapter.model[0].weight, conv_before)
    assert not torch.equal(adapter.model[1].weight, bn_before)
    state = adapter.adaptation_state_dict()
    second = Tent(backbone)
    second.load_adaptation_state_dict(state)
    assert second.n_updates == adapter.n_updates
    assert torch.equal(second.model[1].weight, adapter.model[1].weight)
    adapter.reset()
    assert adapter.n_updates == 0
    assert torch.equal(adapter.model[1].weight, bn_before)


def test_cotta_no_grad():
    torch.manual_seed(2)
    backbone = tiny_bn()
    adapter = CoTTA(backbone, num_augmentations=2, aug_confidence_threshold=1.0,
                    restore_prob=0.0)
    old_student = adapter.model[0].weight.detach().clone()
    old_teacher = adapter.teacher[0].weight.detach().clone()
    with torch.no_grad():
        z = adapter(torch.rand(4,3,16,16))
    assert z.shape == (4,100)
    assert adapter.n_updates == 1
    assert not torch.equal(old_student, adapter.model[0].weight)
    assert not torch.equal(old_teacher, adapter.teacher[0].weight)
    assert torch.equal(adapter.anchor[0].weight, old_student)
    save = adapter.adaptation_state_dict()
    cloned = CoTTA(backbone, num_augmentations=2, aug_confidence_threshold=1.0,
                   restore_prob=0.0)
    cloned.load_adaptation_state_dict(save)
    assert torch.equal(cloned.teacher[0].weight, adapter.teacher[0].weight)
    adapter.reset()
    assert adapter.n_updates == 0


def test_eta_fisher_eata():
    torch.manual_seed(0)
    backbone = tiny_bn()
    fisher = estimate_fisher_from_unlabeled(backbone, [(torch.randn(8,3,16,16),
                                                        torch.randint(100,(8,)))], max_samples=8)
    assert fisher
    for f in fisher.values():
        assert (f >= 0).all()
    eta = ETA(backbone, e_margin=8, d_margin=1)
    with torch.no_grad():
        z = eta(torch.randn(6,3,16,16))
    assert z.shape == (6,100) and eta.n_updates == 1
    eata = EATA(backbone, fisher=fisher, e_margin=8, d_margin=1)
    with torch.no_grad():
        out = eata(torch.randn(6,3,16,16))
    assert out.shape == (6,100) and eata.n_updates == 1


def test_sar():
    torch.manual_seed(0)
    backbone = tiny_gn()
    sar = SAR(backbone, e_margin=8, recovery_entropy_threshold=0.0,
              freeze_final_stage=False)
    before = sar.model[0].weight.detach().clone()
    with torch.no_grad():
        out = sar(torch.randn(6,3,16,16))
    assert out.shape == (6,100) and sar.n_updates == 1
    assert torch.equal(before, sar.model[0].weight)
    assert sar.get_diagnostics()['reliable_second'] == 6
    sar.reset()
    assert sar.n_updates == 0


def test_factory():
    for name in ('source','bn_adapt','tent','cotta','eta'):
        assert create_adapter(name,tiny_bn(),**({'num_augmentations':2} if name=='cotta' else {}))


def test_no_adaptation_on_uncertain_eta():
    torch.manual_seed(5)
    eta = ETA(tiny_bn(), e_margin=0.001)
    weight = eta.model[1].weight.detach().clone()
    with torch.no_grad():
        out = eta(torch.randn(4, 3, 16, 16))
    assert out.shape == (4,100) and eta.n_updates == 0
    assert torch.equal(weight, eta.model[1].weight)
    assert eta.prob_ema is None


def test_cotta_source_restoration_probability_one():
    torch.manual_seed(5)
    adapter = CoTTA(tiny_bn(), restore_prob=1.0,
                    aug_confidence_threshold=0.0, num_augmentations=2)
    assert adapter.n_updates == 0
    with torch.no_grad():
        adapter(torch.rand(4,3,16,16))
    for p, source in zip(adapter.model.parameters(), adapter.anchor.parameters()):
        assert torch.equal(p, source)
    assert adapter.n_updates == 1


def test_sar_uncertain_no_update():
    sar = SAR(tiny_gn(), e_margin=0.0, freeze_final_stage=False)
    before = sar.model[1].weight.detach().clone()
    with torch.no_grad():
        out = sar(torch.randn(4,3,16,16))
    assert out.shape == (4,100)
    assert sar.n_updates == 0
    assert torch.equal(before, sar.model[1].weight)


def test_bn_ema_only_buffers_change_after_eval_called():
    bn = BatchNormAdapt(tiny_bn(), mode="ema").eval()
    before = bn.model[1].running_mean.detach().clone()
    with torch.no_grad():
        out = bn(torch.randn(4,3,16,16)+4)
    assert out.shape == (4,100)
    assert not torch.equal(before, bn.model[1].running_mean)
    state_after = bn.model[1].running_mean.detach().clone()
    bn.predict_only(torch.randn(4,3,16,16))
    assert torch.equal(state_after, bn.model[1].running_mean)
    bn.reset()
    assert torch.equal(before, bn.model[1].running_mean)


def test_trainable_model_grad_and_strict_reload():
    model = create_model('resnet20_cifar', num_classes=100)
    image = torch.randn(2,3,32,32)
    loss = F.cross_entropy(model(image),torch.tensor([0,1]))
    loss.backward()
    assert model.fc.weight.grad is not None
    state=copy.deepcopy(model.state_dict())
    model2=create_model('resnet20_cifar',num_classes=100).eval()
    model2.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(image),model2(image), atol=0,rtol=0)
