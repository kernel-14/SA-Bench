import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics import InceptionScore

def evaluate_model(model, dataloader, device):
    model.eval()
    fid = FrechetInceptionDistance()
    kid = KernelInceptionDistance(subset_size=50)
    inception_score = InceptionScore()

    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            generated_images = model(x, sigma=torch.tensor(0.1).to(device))

            # Update metrics
            fid.update(generated_images, real=False)
            fid.update(x, real=True)

            kid.update(generated_images, real=False)
            kid.update(x, real=True)

            inception_score.update(generated_images)

    fid_score = fid.compute()
    kid_score = kid.compute()
    inception_score_value = inception_score.compute()

    print(f"FID: {fid_score}")
    print(f"KID: {kid_score}")
    print(f"Inception Score: {inception_score_value}")

    return fid_score, kid_score, inception_score_value