import torch
import torch.nn as nn

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc
def run_bdcspn(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    import baselines.proto_rect as lib

    kwargs = {"feature_normalization": 2}  # default 2
    device = clip_weights.device
    fewshot_model = lib.BDCSPN(nn.Identity(), **kwargs).to(device)

    print("\n-------- Searching hyperparameters on the val set. --------")
    neighbor_index = torch.argmax(val_features @ test_features.T, dim=1)

    val_labels = val_labels.cuda()
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
    # Zero-shot CLIP

    Z_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    epsilon_list = [2.5, 5, 10, 20, 40]
    best_acc = -1
    best_Z = 0
    best_epsilon = 0
    for Z in Z_list:
        for epsilon in epsilon_list:
            fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
            method_logits = fewshot_model(test_features, Z=Z, epsilon=epsilon).detach().data

            acc = cls_acc(method_logits[neighbor_index, :], val_labels)
            print("Z = {} ; epsilon = {}".format(Z, epsilon))
            print("**** BDCSPN's val accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_Z = Z
                best_epsilon = epsilon


    print("\n-------- Evaluating on the test set. --------")


    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
    method_logits = fewshot_model(test_features, Z=best_Z, epsilon=best_epsilon).detach().data

    acc = cls_acc(method_logits, test_labels)
    print("**** bdcspn's test accuracy for seed {}: {:.2f}. ****\n".format(int(cfg["seed"]), acc))
    print("with Z = {} : epsilon = {}  for val accuracy of {:.2f}".format(best_Z, best_epsilon, best_acc))

    return None

def run_tim(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, version="adm"):
    # clip_weights can be ignored
    import baselines.tim as lib

    kwargs_adm = {
      "fine_tuning_steps": 150,
      "cross_entropy_weight": 0.1,
      "marginal_entropy_weight": 1.0,
      "conditional_entropy_weight": 0.1,
      "temperature": 15.0,
      "alpha": 1.0,
      "feature_normalization": 2
    }
    kwargs_gd = {
        "fine_tuning_steps": 1000,
        "fine_tuning_lr": 0.0001,
        "cross_entropy_weight": 0.1,
        "marginal_entropy_weight": 1.0,
        "conditional_entropy_weight": 0.1,
        "temperature": 15,
        "feature_normalization": 2
    }
    # Warning : difference between github and paper


    device = clip_weights.device
    if version=="gd":
        fewshot_model = lib.TIM(nn.Identity(), **kwargs_gd).to(device)
        word = "gd"
    else:
        fewshot_model = lib.TIM_ADM_bis(nn.Identity(), **kwargs_adm).to(device)
        word = "adm"

    print("\n-------- Searching hyperparameters on the val set. --------")
    neighbor_index = torch.argmax(val_features @ test_features.T, dim=1)
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
    val_labels = val_labels.cuda()

    temperature_list = [5, 10, 15, 30, 60]
    xentr_list = [0.025, 0.05, 0.1, 0.2, 0.4]
    centr_list = [0.025, 0.05, 0.1, 0.2, 0.4]
    best_acc = -1
    best_t = 0
    best_xentr = 0
    best_centr = 0
    for temp in temperature_list:
        for xentr in xentr_list:
            for centr in centr_list:
                kwargs_adm["temperature"] = temp
                kwargs_adm["cross_entropy_weight"] = xentr
                kwargs_adm["conditional_entropy_weight"] = centr
                fewshot_model = lib.TIM_ADM_bis(nn.Identity(), **kwargs_adm).to(device)

                fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
                method_logits = fewshot_model(test_features).detach().data
                acc = cls_acc(method_logits[neighbor_index, :], val_labels)
                print("temperature = {} ; xentr = {}  ; centr = {}".format(temp, xentr, centr))
                print("**** TIM's val accuracy: {:.2f}. ****\n".format(acc))
                if acc > best_acc:
                    best_acc = acc
                    best_t = temp
                    best_xentr = xentr
                    best_centr = centr


    print("\n-------- Evaluating on the test set. --------")
    kwargs_adm["temperature"] = best_t
    kwargs_adm["cross_entropy_weight"] = best_xentr
    kwargs_adm["conditional_entropy_weight"] = best_centr
    fewshot_model = lib.TIM_ADM_bis(nn.Identity(), **kwargs_adm).to(device)


    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
    method_logits = fewshot_model(test_features).detach().data

    acc = cls_acc(method_logits, test_labels)
    print("**** tim_{}'s test accuracy for seed {}: {:.2f}. ****\n".format(word, int(cfg["seed"]), acc))
    print("with temperature = {} ; xentr = {}  ; centr = {}  ; best val acc = {:.2f}".format(best_t, best_xentr, best_centr, best_acc))

    return None


def run_laplacian_shot(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    # clip_weights can be ignored
    import baselines.laplacian_shot as lib

    kwargs = {
        "inference_steps": 20,
        "knn": 3,
        "lambda_regularization": 0.7,
        "feature_normalization": 2
    }

    device = clip_weights.device

    print("\n-------- Searching hyperparameters on the val set. --------")
    neighbor_index = torch.argmax(val_features @ test_features.T, dim=1)
    val_labels = val_labels.cuda()
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    knn_list = [3, 5, 10]
    lambda_list = [0.1, 0.3, 0.5, 0.7, 0.8, 1, 1.2, 1.5]
    best_acc = -1
    best_k = 0
    best_l = 0
    for k in knn_list:
        for l in lambda_list:
            kwargs["knn"] = k
            kwargs["lambda_regularization"] = l
            fewshot_model = lib.LaplacianShot(nn.Identity(), **kwargs).to(device)
            fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
            method_logits = fewshot_model(test_features).detach().data
            acc = cls_acc(method_logits[neighbor_index, :], val_labels)
            print("k = {} ; lambda = {}".format(k, l))
            print("**** Laplacian shot's val accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_l = l


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    kwargs["knn"] = best_k
    kwargs["lambda_regularization"] = best_l
    fewshot_model = lib.LaplacianShot(nn.Identity(), **kwargs).to(device)
    fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
    method_logits = fewshot_model(test_features).detach().data

    acc = cls_acc(method_logits, test_labels)
    print("**** laplacian_shot's test accuracy for seed {}: {:.2f}. ****\n".format( int(cfg["seed"]), acc))
    print("with k = {} : lambda = {}  for val accuracy of {:.2f}".format(best_k, best_l, best_acc))

    return None


def run_transductive_finetuning(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    # clip_weights can be ignored
    import baselines.transductive_finetuning as lib

    kwargs = {
        "fine_tuning_steps": 25,
        "fine_tuning_lr":  5e-5,
        "temperature": 1.0,
        "feature_normalization": 2
    }

    device = clip_weights.device
    fewshot_model = lib.TransductiveFinetuning(nn.Identity(), **kwargs).to(device)

    print("\n-------- Searching hyperparameters on the val set. --------")
    neighbor_index = torch.argmax(val_features @ test_features.T, dim=1)
    val_labels = val_labels.cuda()
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    t_list = [0.25, 0.5, 1, 2, 4]
    step_list = [10, 15, 20, 25, 30, 35, 40]
    best_acc = -1
    best_t = 0
    best_s = 0
    for t in t_list:
        for s in step_list:
            kwargs["fine_tuning_steps"] = s
            kwargs["temperature"] = t
            fewshot_model = lib.TransductiveFinetuning(nn.Identity(), **kwargs).to(device)
            fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
            method_logits = fewshot_model(test_features).detach().data
            acc = cls_acc(method_logits[neighbor_index, :], val_labels)
            print("temp = {} ; steps = {}".format(t, s))
            print("**** Transductive finetuning's val accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_t = t
                best_s = s


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    kwargs["fine_tuning_steps"] = best_s
    kwargs["temperature"] = best_t
    fewshot_model = lib.TransductiveFinetuning(nn.Identity(), **kwargs).to(device)
    fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
    method_logits = fewshot_model(test_features).detach().data

    acc = cls_acc(method_logits, test_labels)
    print("**** transductive_finetuning's test accuracy for seed {}: {:.2f}. ****\n".format(int(cfg["seed"]), acc))
    print("with temp = {} : steps = {}  for val accuracy of {:.2f}".format(best_t, best_s, best_acc))

    return None

def run_pt_map(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    # clip_weights can be ignored
    import baselines.pt_map as lib

    kwargs = {
        "fine_tuning_steps": 10,
        "fine_tuning_lr":  0.2,
        "lambda_regularization": 10.0,
        "power_factor": 0.5,
        "feature_normalization": 2
    }

    device = clip_weights.device
    fewshot_model = lib.PTMAP(nn.Identity(), **kwargs).to(device)

    print("\n-------- Searching hyperparameters on the val set. --------")
    neighbor_index = torch.argmax(val_features @ test_features.T, dim=1)
    val_labels = val_labels.cuda()
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    lr_list = [0.2, 0.4]
    best_acc = -1
    for lr in lr_list:

        kwargs["fine_tuning_lr"] = lr
        fewshot_model = lib.PTMAP(nn.Identity(), **kwargs).to(device)
        fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
        method_logits = fewshot_model(test_features).detach().data
        acc = cls_acc(method_logits[neighbor_index, :], val_labels)
        print("alpha = {} ".format(lr))
        print("**** PTMAP's val accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_lr = lr


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    kwargs["fine_tuning_lr"] = best_lr
    fewshot_model = lib.PTMAP(nn.Identity(), **kwargs).to(device)
    fewshot_model.process_support_set(cache_keys, cache_values.argmax(dim=1))
    method_logits = fewshot_model(test_features).detach().data

    acc = cls_acc(method_logits, test_labels)
    print("**** ptmap's test accuracy for seed {}: {:.2f}. ****\n".format(int(cfg["seed"]), acc))
    print("with alpha = {}  for val accuracy of {:.2f}".format(best_lr, best_acc))

    return None