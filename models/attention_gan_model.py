import torch
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import itertools
from torchsummary import summary
import torch_optimizer as optimizer
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import PerceptualLoss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .VGGPerceptualLoss import VGGLoss,VGGNet
from .losses import PerceptualLoss  as P
class AttentionGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser
     
    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'o1_b', 'o2_b', 'o3_b', 'o4_b', 'o5_b', 'o6_b', 'o7_b', 'o8_b', 'o9_b', 'o10_b',
        'a1_b', 'a2_b', 'a3_b', 'a4_b', 'a5_b', 'a6_b', 'a7_b', 'a8_b', 'a9_b', 'a10_b', 'i1_b', 'i2_b', 'i3_b', 'i4_b', 'i5_b', 
        'i6_b', 'i7_b', 'i8_b', 'i9_b']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'o1_a', 'o2_a', 'o3_a', 'o4_a', 'o5_a', 'o6_a', 'o7_a', 'o8_a', 'o9_a', 'o10_a', 
        'a1_a', 'a2_a', 'a3_a', 'a4_a', 'a5_a', 'a6_a', 'a7_a', 'a8_a', 'a9_a', 'a10_a', 'i1_a', 'i2_a', 'i3_a', 'i4_a', 'i5_a', 
        'i6_a', 'i7_a', 'i8_a', 'i9_a']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'a10_b', 'real_B','fake_A', 'a10_a']
        else:
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        nb = opt.batchSize
        size = opt.fineSize
        
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        summary(self.netG_A, (3, 256, 256))

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netFeat = networks.define_feature_network(opt.which_model_feat, self.gpu_ids)
       # summary(self.netD_B, (3, 256, 256))

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCDGAN = networks.GANLoss2(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCS = torch.nn.L1Loss()
            self.CS = torch.nn.L1Loss()
            self.criterionFeat = PerceptualLoss(nn.MSELoss())
            self.criterionSyn = torch.nn.L1Loss()
            #self.criterionCS = torch.nn.L1Loss()
            self.ssim_module = SSIM(data_range=255, size_average=True, channel=3)
            self.ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = optimizer.DiffGrad(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = optimizer.DiffGrad(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        #print("******realA******")
        #print(self.real_A)
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.o1_b, self.o2_b, self.o3_b, self.o4_b, self.o5_b, self.o6_b, self.o7_b, self.o8_b, self.o9_b, self.o10_b, \
        self.a1_b, self.a2_b, self.a3_b, self.a4_b, self.a5_b, self.a6_b, self.a7_b, self.a8_b, self.a9_b, self.a10_b, \
        self.i1_b, self.i2_b, self.i3_b, self.i4_b, self.i5_b, self.i6_b, self.i7_b, self.i8_b, self.i9_b = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A, _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _ = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A, self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a, \
        self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a, \
        self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B, _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _ = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        x =torch.mean(real)
        #print((x.size))
        #assert False
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
 #################RALSGAN###############################
        
        loss_D1 = (torch.mean((pred_real - torch.mean(pred_fake) - 1) ** 2) + torch.mean((pred_fake - torch.mean(pred_real) + 1) ** 2))/2

        # Combined loss and calculate gradients
        ###################loss_D = (loss_D_real + loss_D_fake) * 0.5 
        loss_D = loss_D1 + (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_feat_AfA = self.opt.lambda_feat_AfA    
        lambda_feat_BfB = self.opt.lambda_feat_BfB

        lambda_feat_fArecA = self.opt.lambda_feat_fArecA
        lambda_feat_fBrecB = self.opt.lambda_feat_fBrecB

        lambda_feat_ArecA = self.opt.lambda_feat_ArecA
        lambda_feat_BrecB = self.opt.lambda_feat_BrecB
        
        lambda_syn_A = self.opt.lambda_syn_A
        lambda_syn_B = self.opt.lambda_syn_B
        
        lambda_CS_A = self.opt.lambda_CS_A
        lambda_CS_B = self.opt.lambda_CS_B
        
  
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _  = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _  = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        #########RALSGAN#########
        #########
        #########
        #pred_real = netD(real)
        #pred_fake = netD(fake.detach())
        self.loss_G1 = (torch.mean((self.real_B - torch.mean(self.fake_B) + 1) ** 2) + torch.mean((self.fake_B - torch.mean(self.real_B) - 1) ** 2))/2
        self.loss_G2 = (torch.mean((self.real_A - torch.mean(self.fake_A) + 1) ** 2) + torch.mean((self.fake_A - torch.mean(self.real_A) - 1) ** 2))/2
        
       
        ########################################
        #k = 15
        ##selfSA= self.criterionCS(self.fake_A,self.real_A.detach())*k
        ##self.CSB= self.criterionCS(self.fake_B,self.real_B.detach())*k
        #new LOSS
        self.loss_CSA = self.criterionCS(self.fake_A,self.rec_A)*lambda_CS_A
        self.loss_CSB = self.criterionCS(self.fake_B,self.rec_B)*lambda_CS_A
        self.loss_CS_A = self.criterionCS(self.fake_A,self.real_A)*lambda_CS_A
        self.loss_CS_B = self.criterionCS(self.fake_B,self.real_B)*lambda_CS_A
        self.loss_CS_A1 = self.criterionCS(self.rec_A,self.real_A)*lambda_CS_A
        self.loss_CS_B1 = self.criterionCS(self.rec_B,self.real_B)*lambda_CS_A

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
         # Synthesized loss
        self.loss_SynB = self.criterionSyn(self.fake_B, self.real_B) *lambda_syn_B
        self.loss_SynA = self.criterionSyn(self.fake_A, self.real_A) *lambda_syn_A
       
        self.loss_AfA = 0#1 - self.ssim_module(self.real_A, self.fake_A)*lambda_CS_B    
        self.loss_BfB =0# 1 - self.ssim_module(self.real_B, self.fake_B)*lambda_CS_B 
        self.loss_AfAm = 1 - self.ms_ssim_module(self.real_A, self.fake_A)*lambda_CS_B    
        self.loss_BfBm = 1 - self.ms_ssim_module(self.real_B, self.fake_B)*lambda_CS_B 
######################################################################################################################################################
        lambda_feat_ArecA= 1
        lambda_feat_BrecB =1
        lambda_feat_AfA =1 
        lambda_feat_BfB =1
        lambda_feat_fArecA =1
        lambda_feat_fBrecB =1


        self.feat_loss_AfA = self.criterionFeat.get_loss(self.fake_A, self.real_A) * lambda_feat_AfA    
        self.feat_loss_BfB = self.criterionFeat.get_loss(self.fake_B, self.real_B) * lambda_feat_BfB

        self.feat_loss_fArecA = self.criterionFeat.get_loss(self.fake_A, self.rec_A) * lambda_feat_fArecA
        self.feat_loss_fBrecB = self.criterionFeat.get_loss(self.fake_B, self.rec_B) * lambda_feat_fBrecB

        self.feat_loss_ArecA = self.criterionFeat.get_loss(self.rec_A, self.real_A) * 1#lambda_feat_ArecA 
        self.feat_loss_BrecB = self.criterionFeat.get_loss(self.rec_B, self.real_B  ) *1# lambda_feat_BrecB

        self.feat_loss2 = self.feat_loss_AfA + self.feat_loss_BfB+ self.feat_loss_fArecA+ self.feat_loss_fBrecB + self.feat_loss_ArecA + self.feat_loss_BrecB 

        self.syn_loss =  self.loss_SynB + self.loss_SynA  +self.loss_CS_A+ self.loss_CS_B + self.loss_CSA +  self.loss_CSB +self.loss_CS_A1 +self.loss_CS_B1 +self.loss_AfA+ self.loss_BfB + self.loss_AfAm + self.loss_BfBm + self.feat_loss2 + self.loss_G1  +self.loss_G2

        self.rec_A = self.netG_B(self.fake_B)


        self.rec_B = self.netG_A(self.fake_A)


        
        self.rec_B = self.netG_A.forward(self.fake_A)

        self.loss_G = self.loss_G_A + self.loss_G_B  + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B  + self.syn_loss 
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
    def get_current_errors(self):
        D_A = self.loss_D_A.data.item()
        G_A = self.loss_G_A.data.item()
        Cyc_A = self.loss_cycle_A.data.item()
        D_B = self.loss_D_B.data.item()
        G_B = self.loss_G_B.data.item()
        Cyc_B = self.loss_cycle_B.data.item()
        CSA = self.loss_CSA.data.item()
        CSB = self.loss_CSB.data.item()
        SynA = self.loss_SynA.data.item()
        SynB = self.loss_SynB.data.item()
        
        
        
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data.item()
            idt_B = self.loss_idt_B.data.item()
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B), 
#                                ('D_A1', D_A1), ('G_A1', G_A1), ('D_B1', D_B1), ('G_B1', G_B1), 
                                ('CSA', CSA), ('CSB', CSB), ('SynA', SynA), ('SynB', SynB) ])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),
#                                ('D_A1', D_A1), ('G_A1', G_A1), ('D_B1', D_B1), ('G_B1', G_B1), 
                                ('CSA', CSA), ('CSB', CSB), ('SynA', SynA), ('SynB', SynB) ])
    


     
