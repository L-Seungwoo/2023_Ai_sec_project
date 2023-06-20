import torchvision
import os
import torch
from PIL import Image
from torchvision import transforms

## image transformation
transform = torchvision.transforms.Compose([            #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225])                 #[7]
])



## multi-sample에서도 동작하지만, 각 input에 대한 saliency를 원하므로 원래 용도는 단일 sample input에 대해 사용
def saliency_profile(net, input_tensor, ground_truth_label):
    
    net.eval()
    
    filter_grads = []
    
    loss_func = torch.nn.CrossEntropyLoss()
    
    label = ground_truth_label
    
    if not type(label)==list:
        label = [label]
    if not type(label) == torch.Tensor:
        label = torch.Tensor(label).type(torch.long)
    
    output_tensor = net(input_tensor)
    loss = loss_func(output_tensor, label)
    
    ## Manually .zero_grad()
    for param in net.parameters():
        if not param.grad==None:
            param.grad.data.zero_()
    
    gradients = torch.autograd.grad(loss, net.parameters(), create_graph=True, allow_unused = True)
    
    for i in range(len(gradients)):
        if len(gradients[i].size()) == 4:
            filter_grads.append(gradients[i].abs().mean(-1).mean(-1).mean(-1))
    
    return gradients, filter_grads

## Compute mean and std of saliency of testset
def testset_saliency_mean_and_std(net, input_tensors, ground_truth_labels):
    
    n_samples = len(input_tensors)
    print("Number of test samples : ", n_samples)
    
    _, mean = saliency_profile(net, input_tensors[0].unsqueeze(dim=0), ground_truth_labels[0])
    for i in range(1, n_samples):
        _, filter_grads = saliency_profile(net, input_tensors[i].unsqueeze(dim=0), ground_truth_labels[i])
        mean = [mean[j]+filter_grads[j] for j in range(len(mean))]
    mean = [mean[i]/n_samples for i in range(len(mean))]
    
    _, std = saliency_profile(net, input_tensors[0].unsqueeze(dim=0), ground_truth_labels[0])
    std = [(std[i]-mean[i])**2 for i in range(len(mean))]
    
    for i in range(1, n_samples):
        _, filter_grads = saliency_profile(net, input_tensors[i].unsqueeze(dim=0), ground_truth_labels[i])
        std = [std[j] + (std[j]-mean[j])**2 for j in range(len(mean))]
    std = [(std[i]/n_samples)**(1/2) for i in range(len(mean))]
    
    
    return mean, std

def standardize_saliency_profile(net, filter_grads, mean, std):
    saliency_profile = torch.cat(filter_grads)
    mean_tensor = torch.cat(mean)
    std_tensor = torch.cat(std)
    std_tensor[std_tensor <= 1e-14] = 1
    
    saliency_profile = (saliency_profile-mean_tensor)/std_tensor
    
    return saliency_profile

# top100_salient_filters = torch.topk(saliency_profile, 100)

def measure_top1_accuracy(net, path, ground_truth_list):
    net.eval()
    ctr = 0
    incorrect_ctr = 0
    
    image_list = os.listdir(path) # import os
    image_list = image_list[:100]
    for index, img_file_name in enumerate(image_list):
        img = Image.open(path+'/'+img_file_name)
        if not img.mode == 'RGB':
            continue
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        net(batch_t)
        if not torch.argmax(net(batch_t).squeeze()).item() == ground_truth_list[index]:
            incorrect_ctr += 1
        
        ctr += 1
    print("total samples : ", ctr)
    print("misclassified samples : ", incorrect_ctr)
    return 1-incorrect_ctr/float(ctr)

def find_n_mis_misimages(net, path, ground_truth_list, n_mis):
    
    net.eval()
    
    mis_img_tensor = torch.empty((0,3,224,224), dtype=torch.float32)
    misclassified_imgs_list = []
    mis_ctr = 0
    
    image_list = os.listdir(path)
    image_list = image_list[:1000]
    for index, img_file_name in enumerate(image_list):
        if mis_ctr == n_mis: break
        img = Image.open(path+'/'+img_file_name)
        if not img.mode == 'RGB':
            continue
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        

        if not torch.argmax(net(batch_t).squeeze()).item() == ground_truth_list[index]:
            misclassified_imgs_list.append(img_file_name)
            mis_img_tensor = torch.cat((mis_img_tensor, batch_t), dim=0)
            mis_ctr += 1
    
    #print(misclassified_imgs_list)
    return misclassified_imgs_list

def true_class_confidence(net, input, true_class_label):
    output = net(input).squeeze()
    confidence = torch.nn.functional.softmax(output, dim=0)[true_class_label].item()*100
    
    return confidence

def experiments(ex_batch_t, ground_truth, testset_mean, testset_std, path_testset, ground_truth_list_testset, k, theta, confidence_score_list, accuracy_list):
    net = torchvision.models.resnet50(pretrained=True)
    net.eval()

    ex_grad, ex_saliency = saliency_profile(net, ex_batch_t, ground_truth)
    ex_standardized_saliency =  standardize_saliency_profile(net, ex_saliency, testset_mean, testset_std)

    ex_topk_salient_filters = torch.topk(ex_standardized_saliency, k)

    #print(true_class_confidence(net=net, input=ex_batch_t, true_class_label=ground_truth_list_find_mis[1]))

    ex_output = net(ex_batch_t).squeeze()
    ex_output[ground_truth].backward(create_graph=True)

    ex_total_filter_index = 0


    for layers in net.parameters():
        ex_local_filter_index = 0

        if len(layers.size()) == 4 :
            for filters in layers:
                if ex_total_filter_index in ex_topk_salient_filters.indices:
                    #print("ex_local_filter_index: ", ex_local_filter_index)
                    #print("ex_total_filter_index: ", ex_total_filter_index)
                    #print("layers.size(): ", layers.size())
                    #print("layers.grad.size(): ", layers.grad.size())
                    #print("\n")
                    tmp1 = layers[ex_local_filter_index]
                    tmp2 = layers.grad[ex_local_filter_index]
                    layers[ex_local_filter_index] = tmp1+theta*tmp2


                ex_local_filter_index += 1 
                ex_total_filter_index += 1

    confidence_score_list.append(true_class_confidence(net=net, input=ex_batch_t, true_class_label=ground_truth))
    accuracy_list.append(measure_top1_accuracy(net, path_testset, ground_truth_list_testset))
    #print(true_class_confidence(net=net, input=ex_batch_t, true_class_label=ground_truth_list_find_mis[32]))



## Load resnet50 pretrained
## eval or train ?
net = torchvision.models.resnet50(pretrained=True)



## Prepare ground truth labels of the data
ground_truth_list_find_mis = []
ground_truth_list_testset = []

with open('/mnt/d/caffe_ilsvrc12/val.txt', 'r') as f:
    lines = f.read().splitlines()
    for i in range(1000):
        label = int(lines[i].split()[1])
        ground_truth_list_find_mis.append(label)
    for i in range(10000,11000):
        label = int(lines[i].split()[1])
        ground_truth_list_testset.append(label)
        
## File path to measure accuracy
path_testset = "./data/ILSVRC2012_img_val_testset"

## File path to find misclassified images
path_find_mis = "./data/ILSVRC2012_img_val_find_mis"

## Prepare misclassified image # For multiple misclassified images, use find_n_mis_misimages function
our_mis_img_fname = 'ILSVRC2012_val_00000033.JPEG'
our_true_label = ground_truth_list_find_mis[32]




## Compute top-1 accuracy of pretrained data on testset
accuracy_before_update = measure_top1_accuracy(net, path_testset, ground_truth_list_testset)

## Sample n_mis datas that are misclassified 
#n_mis = 100
#mis_img_filename_list = find_n_mis_misimages(net, path_find_mis, ground_truth_list_find_mis, n_mis)


testset_tensors = torch.empty((0,3,224,224), dtype=torch.float32)
testset_tensors_ctr = 0
path_testset_list = os.listdir(path_testset)

for image_file_name in path_testset_list:
    if testset_tensors_ctr == 10: break
    img = Image.open(path_testset+'/'+image_file_name)
    if not img.mode == 'RGB':
        continue
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    testset_tensors = torch.cat((testset_tensors, batch_t), dim=0)
    testset_tensors_ctr += 1




testset_mean, testset_std = testset_saliency_mean_and_std(net=net, input_tensors=testset_tensors, ground_truth_labels=ground_truth_list_testset)



### test with a misclassified sample
ex_img_file_name = 'ILSVRC2012_val_00000033.JPEG'
ex_img = Image.open(path_find_mis+'/'+ex_img_file_name)
ex_img_t = transform(ex_img)
ex_batch_t = torch.unsqueeze(ex_img_t, 0)



import numpy as np
k0 = 25
theta0 = 0.01

k_list = np.linspace(1, 100, 100)
k_list = [int(k_list[i]) for i in range(40,60)]
theta_list = np.linspace(0.001, 0.1, 100)
theta_list = [float(theta_list[i]) for i in range(20)]

confidence_score_list_theta0 = []
accuracy_list_theta0 = []
confidence_score_list_k0 = []
accuracy_list_k0 = []

for theta in theta_list:
    experiments(ex_batch_t, ground_truth_list_find_mis[32], testset_mean, testset_std, path_testset, ground_truth_list_testset, k0, theta, confidence_score_list_k0, accuracy_list_k0)



print(confidence_score_list_theta0), print(accuracy_list_theta0) # results



## Error occurs for multiple misclassified samples
"""
theta = 0.01
topk = 10

for img_file_names in mis_img_filename_list:
    
    mis_index = int(img_file_names[15:23])-1
    #print("mis_index : ",mis_index)
    mis_img = Image.open(path_find_mis+'/'+img_file_names)
    mis_img_t = transform(mis_img)
    mis_batch_t = torch.unsqueeze(mis_img_t, 0)

    _, mis_saliency = saliency_profile(net, mis_batch_t, ground_truth_list_find_mis[mis_index])
    mis_standardized_saliency =  standardize_saliency_profile(net, mis_saliency, testset_mean, testset_std)

    mis_top_salient_filters = torch.topk(mis_standardized_saliency, topk)

    #print(true_class_confidence(net=net, input=ex_batch_t, true_class_label=ground_truth_list_find_mis[1]))

    mis_output = net(mis_batch_t).squeeze()
    
    ## Manually .zero_grad()
    for param in net.parameters():
        if not param.grad==None:
            param.grad.data.zero_()
    
    mis_output[ground_truth_list_find_mis[mis_index]].backward(create_graph=True)

    mis_total_filter_index = 0

    for layers in net.parameters():
        mis_local_filter_index = 0

        if len(layers.size()) == 4 :
            for filters in layers:
                if mis_total_filter_index in mis_top_salient_filters.indices:
                    #print("ex_local_filter_index: ", ex_local_filter_index)
                    #print("ex_total_filter_index: ", ex_total_filter_index)
                    #print("layers.size(): ", layers.size())
                    #print("layers.grad.size(): ", layers.grad.size())
                    #print("\n")
                    tmp1 = layers[mis_local_filter_index]
                    tmp2 = layers.grad[mis_local_filter_index]
                    layers[mis_local_filter_index] = tmp1+theta*tmp2


                mis_local_filter_index += 1 
                mis_total_filter_index += 1

#print(true_class_confidence(net=net, input=ex_batch_t, true_class_label=ground_truth_list_find_mis[1]))
"""

