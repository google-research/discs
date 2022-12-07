r"""Launcher for running a experiments.

Usage:
xmanager launch \
  discs/run_launcher.py -- \
    --model_config="discs/models/configs/bernoulli_config.py" \
    --sampler_config="discs/samplers/configs/randomwalk_config.py" \
    --xm_resource_alloc="user:xcloud/${USER}"
"""

from absl import app
from absl import flags
import os
import getpass
from copy import deepcopy
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_local
from ml_collections import config_flags

config_flags.DEFINE_config_file(
    name='model_config',
    default=None,
    help_string='Model configuration file.',
    lock_config=True)

config_flags.DEFINE_config_file(
    name='sampler_config',
    default=None,
    help_string='Sampler configuration file.',
    lock_config=True)

flags.DEFINE_string('save_folder_pattern',
                    '/gcs/xcloud-shared/{user}/results/discs/{exp_name}_{exp_id}',
                    'save folder pattern')

_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'samplerrrrrrrr', 'Name of the experiment.', short_name='n')
_LAUNCH_LOCALLY = flags.DEFINE_bool(
    'launch_locally', False,
    'Launch the experiment locally with docker without starting an XManager '
    'experiment.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None,
                                   'Where to save the outputs')
_NUM_GPUS = flags.DEFINE_integer('num_gpus', 2, 'Number of GPUs')
# flags.mark_flags_as_required(['output_path'])
FLAGS = flags.FLAGS


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  job_model_config = FLAGS.model_config
  job_sampler_config = FLAGS.sampler_config
  model_config_filename = config_flags.get_config_filename(FLAGS['model_config'])
  # model_config_file_name = os.path.join('discs/models/configs', model_config_filename)+"_config.py"
  sampler_config_filename = config_flags.get_config_filename(FLAGS['sampler_config'])
  #sampler_config_file_name = os.path.join('discs/models/configs', sampler_config_filename)+"_config.py"
  executable_args = {}
  # Add config flag and related overrides to args.
  executable_args['model_config'] = model_config_filename
  executable_args['sampler_config'] = sampler_config_filename

  uname = getpass.getuser()

  create_experiment = (
      xm_local.create_experiment
      if _LAUNCH_LOCALLY.value else xm_abc.create_experiment)
  with create_experiment(experiment_title=_EXP_NAME.value) as experiment:
    job_requirements = xm.JobRequirements(v100=_NUM_GPUS.value)

    # Creating executor depending on the --launch_locally value.
    executor = (
        xm_local.Local(experimental_stream_output=True)
        if _LAUNCH_LOCALLY.value else xm_abc.Gcp(requirements=job_requirements))

    save_dir = FLAGS.save_folder_pattern.format(
        user=uname, exp_name=_EXP_NAME.value, exp_id=experiment.experiment_id)

    module = 'discs.experiments.main_sampling'
    executable, = experiment.package([
        xm.python_container(
            path='.',
            entrypoint=xm.ModuleName(module),
            use_deep_module=True,
            executor_spec=executor.Spec(),
            args=executable_args)
    ])

    async def make_job(work_unit, **kwargs):
        args = deepcopy(executable_args)
        args.update(kwargs)
        work_unit.add(xm.Job(executable, args=args, executor=executor))

    for sweep_args in job_model_config.get('sweep', [{}]):
        for sweep_sampler_args in job_sampler_config.get('sweep', [{}]):
            sweep_args.update(sweep_sampler_args)
            experiment.add(make_job, args=sweep_args)


if __name__ == '__main__':
  app.run(main)
