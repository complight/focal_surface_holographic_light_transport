import torch
import torch.nn.functional as f
from torchvision.models import vgg16
import torchvision.models as models


# --- Perceptual loss network  --- #
class PER(torch.nn.Module):
    def __init__(self):
        super(PER, self).__init__()
        self.device = torch.device('cuda')
      #  vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg16(pretrained=False).to(self.device)
        pre=torch.load('/home/zheng.chua/freeform_cgh/focal_surface_noise/src/loss/model/vgg16-397923af.pth')
        vgg_model.load_state_dict(pre)
        vgg_model = vgg_model.features[:16].to(self.device)
        # print(vgg_model)
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        x = x.to(self.device)
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        dehaze = dehaze.to(self.device)
        gt = gt.to(self.device)
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(f.l1_loss(dehaze_feature, gt_feature))
        # print('per',sum(loss)/len(loss))
        return sum(loss) / len(loss)

#vgg_model = vgg16(pretrained=True).features[:16]
#vgg_model = vgg_model.to(device)
#for param in vgg_model.parameters():
#    param.requires_grad = False
#loss_network = LossNetwork(vgg_model)
#loss_network.eval()
#s=torch.rand(1,3,512,512)
#v=torch.rand(1,3,512,512)
#a=PER().to()
#s=a(s,v)
#print(s)
#s=a(s,v)
