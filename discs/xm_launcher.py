from absl import app
from absl import flags
import getpass
import numpy as np
import itertools
from copy import deepcopy
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_local
from xmanager.contrib import framework_defaults
from ml_collections import config_flags
import copy
import pdb

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
    name='config',
    default=None,
    help_string=(
        'config containing the model name, sampler and sweep parameters.'
    ),
    lock_config=True,
)

_EXP_NAME = flags.DEFINE_string(
    'experiment_name',
    'Sampling_Experiment',
    'Name of the experiment.',
    short_name='n',
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

  executable_args = {}
  executable_args['model_config'] = (
      f'/workdir/discs/models/configs/{job_config.model}_config.py'
  )
  executable_args['sampler_config'] = (
      f'/workdir/discs/samplers/configs/{job_config.sampler}_config.py'
  )
  
  if job_config.get('graph_type', None):
    executable_args['config'] = (
        f'/workdir/discs/experiments/configs/{job_config.model}/{job_config.graph_type}.py'
    )
    executable_args['model_config.graph_type'] = (f'{job_config.graph_type}')
    executable_args.update({
      'model_config.data_root': '/gcs/xcloud-shared/hadai/data/sco',
    })
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

  with create_experiment(experiment_title=_EXP_NAME.value) as experiment:
    priority = xm.ServiceTier.BATCH if _USE_BATCH.value else xm.ServiceTier.PROD
    job_requirements = xm.JobRequirements(
        ram=8 * FLAGS.num_gpus * xm.GiB,
        cpu=4 * FLAGS.num_gpus,
        v100=FLAGS.num_gpus,
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
        user=uname, exp_name=_EXP_NAME.value, exp_id=experiment.experiment_id
    )
    print('Saving Dir is: ', save_dir)
    executable_args['config.experiment.save_root'] = save_dir
    module = 'discs.experiments.main_sampling'
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
      args = deepcopy(executable_args)
      args.update(kwargs)
      sweep_str_parts = []
      for k, v in kwargs.items():
        if k.startswith('config.'):
          k = k[len('config.') :]
        sweep_str_parts.append(f'{k}={v!r}')
      sweep_str = ','.join(sweep_str_parts)
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
