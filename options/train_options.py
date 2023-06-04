from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=10, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        #parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        #parser.add_argument('--fineSize', type=int, default=256, help='final output size')
        #parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=220, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='wgangp', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]') 
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        ### ADDING LOSSS 
        parser.add_argument('--pan_lambdas', nargs='+', type=float, default=[5.0, 1.0, 1.0, 1.0, 5.0], help='lambdas of PAN_loss')
        #parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        #parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_feat_ArecA', type=float, default=1.0, help='weight for perception loss between real A and reconstructed A ')
        parser.add_argument('--lambda_feat_BrecB', type=float, default=1.0, help='weight for perception loss between real B and reconstruced B ')
        
        
        parser.add_argument('--lambda_syn_A', type=float, default=15.0, help='weight for synthesized loss between real A and fake A ')
        parser.add_argument('--lambda_syn_B', type=float, default=15.0, help='weight for synthesized loss between real B and fake B ')
        parser.add_argument('--lambda_feat_AfA', type=float, default=1.0, help='weight for perception loss between real A and fake A ')
        parser.add_argument('--lambda_feat_BfB', type=float, default=1.0, help='weight for perception loss between real B and fake B ')
        
              
        parser.add_argument('--lambda_CS_A', type=float, default=1.0, help='weight for cyclic-synthesized loss between fake A and reconstructed A ')
        parser.add_argument('--lambda_CS_B', type=float, default=1.0, help='weight for cyclic-synthesized loss between fake B and reconstucted B ')
        parser.add_argument('--lambda_feat_fArecA', type=float, default=0.0, help='weight for perception loss between fake A and reconstructed A ')
        parser.add_argument('--lambda_feat_fBrecB', type=float, default=0.0, help='weight for perception loss between fake B and reconstructed B ')
        

        self.isTrain = True
        return parser
