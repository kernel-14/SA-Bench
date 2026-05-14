from torchvision.models import inception_v3
import torch.nn.functional as F
import torch

def calculate_fid(generated_images, real_images):
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.eval()
    
    gen_features = inception(generated_images).detach()
    real_features = inception(real_images).detach()
    
    fid = F.mse_loss(gen_features, real_features)
    return fid.item()

def calculate_is(generated_images):
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.eval()
    
    batch_logits = inception(generated_images).detach()
    probs = F.softmax(batch_logits, dim=1)
    
    mean_probs = torch.mean(probs, dim=0)
    inception_score = torch.exp(torch.sum(mean_probs * torch.log(mean_probs)))
    return inception_score.item()

