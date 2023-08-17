"""Script used to launch Xmanager."""
import copy
import getpass
import itertools
from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_local
from xmanager.contrib import framework_defaults

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
    name='config',
    default=None,
    help_string=(
        'config containing the model name, sampler and sweep parameters.'
    ),
    lock_config=True,
)
flags.DEFINE_string(
    'save_folder_pattern',
    '/gcs/xcloud-shared/{user}/results/discs/{exp_name}_{exp_id}',
    'save folder pattern',
)

# TODO: Make sure this works when True.
_LAUNCH_LOCALLY = flags.DEFINE_bool(
    'launch_locally',
    False,
    (
        'Launch the experiment locally with docker without starting an XManager'
        ' experiment.'
    ),
)

_NUM_GPUS = flags.DEFINE_integer('num_gpus', 8, 'Number of GPUs')
_USE_BATCH = flags.DEFINE_bool(
    'use_batch', False, 'Enables batch service tier.'
)


def get_sweeps(sweeps):
  cfg = copy.deepcopy(sweeps)
  keys, values = zip(*cfg.items())
  permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
  return permutations_dicts


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  job_config = FLAGS.config
  num_gpus = _NUM_GPUS.value
  executable_args = {}
  # setting model, sampler and experiment config
  executable_args['model_config'] = (
      f'/workdir/discs/models/configs/{job_config.model}_config.py'
  )
  executable_args['sampler_config'] = (
      f'/workdir/discs/samplers/configs/{job_config.sampler}_config.py'
  )
  # co problems
  if job_config.get('graph_type', None):
    executable_args['config'] = (
        f'/workdir/discs/experiment/configs/{job_config.model}/{job_config.graph_type}.py'
    )
    executable_args['model_config.graph_type'] = f'{job_config.graph_type}'
    executable_args['model_config.data_root'] = (
        '/gcs/xcloud-shared/hadai/data/sco'
    )
  # language model
  elif job_config.get('model') == 'text_infilling':
    executable_args['config'] = (
        '/workdir/discs/experiment/configs/lm_experiment.py'
    )
  else:
    executable_args['config'] = '/workdir/discs/common/configs.py'
    num_gpus = 4

  if (
      job_config.get('model') == 'maxcut'
      and job_config.get('graph_type') == 'optsicom'
  ):
    num_gpus = 2
  executable_args.update(
      {
          name: value
          for name, value in FLAGS.flag_values_dict().items()
          if name.startswith('config.')
      }
  )
  create_experiment = (
      xm_local.create_experiment
      if _LAUNCH_LOCALLY.value
      else xm_abc.create_experiment
  )
  exp_name = config_flags.get_config_filename(FLAGS['config']).split('/')
  exp_name = 'discs-' + exp_name[-2] + '-' + exp_name[-1][0:-3]
  with create_experiment(experiment_title=exp_name) as experiment:
    priority = xm.ServiceTier.BATCH if _USE_BATCH.value else xm.ServiceTier.PROD
    job_requirements = xm.JobRequirements(
        ram=8 * num_gpus * xm.GiB,
        cpu=4 * num_gpus,
        v100=num_gpus,
        service_tier=priority,
    )

    # Creating executor depending on the --launch_locally value.
    executor = (
        xm_local.Local(experimental_stream_output=True)
        if _LAUNCH_LOCALLY.value
        else xm_abc.Gcp(requirements=job_requirements)
    )

    uname = getpass.getuser()
    save_dir = FLAGS.save_folder_pattern.format(
        user=uname, exp_name=exp_name, exp_id=experiment.experiment_id
    )
    executable_args['config.experiment.save_root'] = save_dir
    module = 'discs.experiment.main_sampling'
    (executable,) = experiment.package(
        [
            xm.python_container(
                path='.',
                base_image=framework_defaults.base_image(
                    'jax', job_requirements.accelerator
                ),
                entrypoint=xm.ModuleName(module),
                use_deep_module=True,
                executor_spec=executor.Spec(),
                args=executable_args,
            )
        ]
    )

    async def make_job(work_unit, **kwargs):
      args = copy.deepcopy(executable_args)
      args.update(kwargs)
      if 'sampler_config.name' in args.keys():
        args['sampler_config'] = (
            f'/workdir/discs/samplers/configs/{args["sampler_config.name"]}_config.py'
        )

      sweep_str_parts = []
      for k, v in kwargs.items():
        if k.startswith('config.experiment.'):
          k = k[len('config.experiment.') :]
        elif k.startswith('model_config.'):
          k = k[len('model_config.') :]
        elif k.startswith('sampler_config.'):
          k = k[len('sampler_config.') :]
        if isinstance(v, str) and v.startswith('/gcs'):
          splits = v.split('/')
          v = splits[-3] + '/' + splits[-2] + '/' + splits[-1]
        sweep_str_parts.append(f'{k}={v!r}')
      sweep_str = ','.join(sweep_str_parts)
      sweep_str = sweep_str.replace('/', '-')
      args[
          'config.experiment.save_root'
      ] += f'/{work_unit.work_unit_id}_{sweep_str}'
      print('************************')
      print(args)
      print('************************')
      work_unit.add(xm.Job(executable, args=args, executor=executor))

    all_sweeps_configs = []
    list_of_sweep_dicts = job_config.get('sweep', {})
    for sweep_dict in list_of_sweep_dicts:
      sweeps = get_sweeps(sweep_dict)
      all_sweeps_configs.append(sweeps)
      
    all_sweeps_configs = np.hstack(all_sweeps_configs)
    for sweep_args in all_sweeps_configs:
      experiment.add(make_job, args=sweep_args)


if __name__ == '__main__':
  app.run(main)
