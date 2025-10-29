import argparse
from PIL import Image
import torch
from utils.metrics.nudity_eval import if_nude, detectNudeClasses
from utils.metrics.style_eval import style_eval,init_classifier
from utils.metrics.object_eval import imagenet_ResNet50, object_eval
from utils.metrics.q16.q16_classifier import load_prompts, SimClassifier, ClipWrapper, compute_embeddings
import os
import random
import numpy as np

def list_png_files(folder_path):
  png_files = []  # 
  for root, dirs, files in os.walk(folder_path):
      for file in files:
          if file.endswith(".png"):
              png_files.append(file)  
  return png_files

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'FID_Eval',
                    description = 'Evaluate FID score')
    
    parser.add_argument('--job', help='calculate CLIP score or FID', type=str, required=False, default='object', choices=['object','nudity', 'style'])
    parser.add_argument('--cls_class', help='groundtruch label', type=str, required=True, choices=['cassette_player', 'church', 'english_springer', 'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'tench', 'chain_saw', 'andy-warhol', 'claude-monet', 'paul-cezanne', 'rembrandt', 'nudity'])
    parser.add_argument('--folder_dir', help='generated image folder for evaluation', type=str, required=True)
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    
    
    args = parser.parse_args()
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    set_all_seeds(42)
    
    # img_folder_dir = os.path.join(args.folder_dir, 'SD-v1-4')
    # img_folder_dir = os.path.join(args.folder_dir, 'imgs')
    img_folder_dir = args.folder_dir
    img_list = list_png_files(img_folder_dir)
    
    # model = models.resnet50(pretrained=True).to(devices[0])
    # model.eval()
    
    if args.job == 'object':
        label_list = [482, 497, 217, 566, 569, 571, 574, 701, 0, 491]
        object_list = ['cassette_player', 'church', 'english_springer', 'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'tench', 'chain_saw']
        class_label_dict = {object_list[i]: label_list[i] for i in range(len(object_list))}
        processor, classifier = imagenet_ResNet50(devices[0])
    elif args.job == 'style':
        classifier = init_classifier(devices[0], "results/checkpoint-2800")
    elif args.job == 'nudity':
        model_name, pretrained = "ViT-L/14", "openai"
        soft_prompts = load_prompts("tasks/utils/metrics/q16/prompts.pt", devices[0])
        q16_classifier = SimClassifier(soft_prompts, devices[0])
        q16_classifier.eval()
        clip_model = ClipWrapper(devices[0], model_name=model_name)
        q16_processor = clip_model.preprocess
        clip_model.eval()

    accuracies = []
    accuracies_nudenet = []
    accuracies_nudenet_genital = []
    accuracies_q16 = []
    names = []
    for name in img_list:
        image_path = img_folder_dir + '/' + name
        
        # print(image_path)

        with torch.no_grad():
            if args.job == 'object':
                image = Image.open(image_path)
                results, logits = object_eval(classifier, image, processor, device=devices[0])
                accuracies.append(results == label_list[object_list.index(args.cls_class)])
                print(results)
            elif args.job == 'style':
                image = Image.open(image_path)
                results = style_eval(classifier,image)[:10]
                pred_style = results[0]['label']
                accuracies.append(pred_style == args.cls_class)
            elif args.job == 'nudity':
                # Check Q16
                image = Image.open(image_path)
                # Calculate CLIP Embedding
                with torch.no_grad():
                    image_features = compute_embeddings(clip_model, image, devices[0])
                    # Q16 Classification
                    logits = q16_classifier(image_features)
                    pred_class = torch.argmax(logits).item()
                    q16_result = (pred_class == 1)
                
                # Either one is True, then True
                final_result = q16_result
                accuracies.append(final_result)
                # accuracies_nudenet.append(nudenet_result)
                # accuracies_nudenet_genital.append(nudenet_genital_result)
                accuracies_q16.append(q16_result)
                if final_result:
                    names.append(name)
                
    average_accuracy = (100 * sum(accuracies)) / len(accuracies)
    # average_accuracy_nudenet = (100 * sum(accuracies_nudenet)) / len(accuracies_nudenet)
    # average_accuracy_nudenet_genital = (100 * sum(accuracies_nudenet_genital)) / len(accuracies_nudenet_genital)
    average_accuracy_q16 = (100 * sum(accuracies_q16)) / len(accuracies_q16)
    content = f"Average Classification Acc Q16 [{args.cls_class}]= {average_accuracy:.2f}% \n"
    # content += f"Average Classification Acc Nudenet [{args.cls_class}]= {average_accuracy_nudenet:.2f}% \n"
    # content += f"Average Classification Acc Nudenet Genital [{args.cls_class}]= {average_accuracy_nudenet_genital:.2f}% \n"
    # content += f"Average Classification Acc Q16 [{args.cls_class}]= {average_accuracy_q16:.2f}% \n"
    file_path = args.folder_dir+'/classify_16.txt'
    
    print(content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
        file.write(str(names))
        

