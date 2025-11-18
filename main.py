from lib.models import get_model
from util.data.load_data import load_data,load_my_data
from util.optimizers import get_optimizers
from util.train_val import train, run
from util.plotting import init_plot, save_env
from util.logs import init_log, save_checkpoint
import sys
import os
import time
import argparse
import numpy as np
import pickle

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument
arg_parser.add_argument('--dataset', default='mnist', help='data set to train on, cifar10 or mnist')
arg_parser.add_argument('--code_location',default='.', help='where your code is stored')
arg_parser.add_argument('--model_type', default='single_level', help='model type, single_level or hierarchical')
arg_parser.add_argument('--inference_type', default='iterative', help='inference type, standard or iterative')
arg_parser.add_argument('--data_path', default='', help='path to data directory root')
arg_parser.add_argument('--log_path', default='', help='path to log directory root')
args = arg_parser.parse_args()

path_to_config = os.path.join(args.code_location, 'cfg', args.dataset, args.model_type, args.inference_type)
sys.path.insert(0, path_to_config)
from cfg.mnist.single_level.iterative.config import train_config, arch

train_config['data_path'] = args.data_path
train_config['log_root'] = args.log_path
train_config['batch_size'] = 128

log_root = train_config['log_root']
log_path, log_dir = init_log(log_root, train_config)
print('Experiment: ' + log_dir)

#global vis
#vis, handle_dict = init_plot(train_config, arch, env=log_dir)

# load data, labels
data_path = train_config['data_path']
train_config['cuda_device'] = 'cuda'
if ('celeba' in train_config['dataset'].lower()) or ('mnist' in train_config['dataset'].lower()) or ('finch' in train_config['dataset'].lower()):
    print('loading using my data')
    train_loader,val_loader,label_names = load_my_data(train_config['dataset'],data_path,train_config['batch_size'],cuda_device= train_config['cuda_device'])
else:
    print("tried to load from somewhere else!")
    print(train_config['dataset'])
    assert False
    train_loader, val_loader, label_names = load_data(train_config['dataset'], data_path, train_config['batch_size'],
                                                    cuda_device=train_config['cuda_device'])

# construct model

for n_iterations in [1,5,50,100,200,300]:
    save_path = os.path.join(args.log_path,f'n_iterations_{n_iterations}_results.pkl')
    train_config['n_iterations'] = n_iterations
    model = get_model(train_config, arch, train_loader)
    iter_dict = {'elbos':{'train':[],'val':[]},
                 'log_probs':{'train':[],'val':[]},
                 'kls':{'train':[],'val':[]}}
    # get optimizers
    (enc_opt, enc_scheduler), (dec_opt, dec_scheduler), start_epoch = get_optimizers(train_config, arch, model)


    for epoch in range(start_epoch+1,300):

        tic = time.time()
        model.train()
        train_dict = train(model, train_config, arch, train_loader, epoch+1, (enc_opt, dec_opt))
        iter_dict['elbos']['train'].append(train_dict['avg_elbo'])
        iter_dict['log_probs']['train'].append(train_dict['avg_cond_like'])
        iter_dict['kls']['train'].append(train_dict['avg_kl'][-1])
        toc = time.time()
        print('Training Time: ' + str(toc - tic))
        # validation
        tic = time.time()
        visualize = False
        eval = False
        if epoch % train_config['display_iter'] == train_config['display_iter']-1:
            save_checkpoint(model, (enc_opt, dec_opt), epoch)
            visualize = True
        if epoch % train_config['eval_iter'] == train_config['eval_iter']-1:
            eval = True
        model.eval()
        output_dict = run(model, train_config, arch, val_loader, vis=visualize, eval=eval)
        average_elbo = np.mean(output_dict['total_elbo'][:, -1], axis=0)
        average_cond_log_like = np.mean(output_dict['total_cond_log_like'][:, -1], axis=0)
        average_kl = [0. for _ in range(len(output_dict['total_kl']))]
        for level in range(len(output_dict['total_kl'])):
            average_kl[level] = np.mean(output_dict['total_kl'][level][:, -1], axis=0)
        averages = average_elbo, average_cond_log_like, average_kl
        toc = time.time()
        print('Validation Time: ' + str(toc - tic))
        print('ELBO: ' + str(averages[0]))
        #save_env()
        enc_scheduler.step()
        dec_scheduler.step()
        iter_dict['elbos']['val'].append(output_dict['total_elbo'][:,-1])
        iter_dict['log_probs']['val'].append(output_dict['total_cond_log_like'][:,-1])
        iter_dict['kls']['val'].append(output_dict['total_kl'][0][:,-1])

    with open(save_path,'wb') as f:
        pickle.dump(iter_dict,f)
    
