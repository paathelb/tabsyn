import torch
from utils import execute_function, get_args

if __name__ == '__main__':
    
    args = get_args()
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'

    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}/{args.method}.csv'
    import pdb; pdb.set_trace()
    main_fn = execute_function(args.method, args.mode)  # "vae", "train"
    
    main_fn(args)