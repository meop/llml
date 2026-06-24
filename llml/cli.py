import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import click
from git import GitCommandNotFound

from llml.actions import (
  app_version,
  executable_version,
  remove_models,
  serve_instance,
  sync_models,
  tidy_model_dir,
)
from llml.repos import refresh_config_repo
from llml.errors import CliError
from llml.instances import all_models, find_instances, installed_instances, list_instances, load_instance
from llml.settings import Settings, load_settings


@dataclass
class State:
  settings: Settings
  dry_run: bool


def print_instances_help(action: str, settings: Settings) -> None:
  suffix = ' [model ...]' if action in {'sync', 'remove'} else ''
  click.echo(f'usage: llml {action} <instance>{suffix}')
  click.echo()
  click.echo('instances:')
  for name in list_instances(settings):
    click.echo(f'  {name}')


def print_models_help(action: str, instance_name: str, instance: dict) -> None:
  suffix = ' [model ...]' if action in {'sync', 'remove'} else ''
  click.echo(f'usage: llml {action} {instance_name}{suffix}')
  click.echo()
  click.echo('models:')
  for name in all_models(instance):
    click.echo(f'  {name}')


def state_from_context(ctx: click.Context) -> State:
  return ctx.find_root().obj


def echo_matches(matches: list[tuple[str, list[str]]]) -> None:
  for name, model_names in matches:
    click.echo(name)
    for model_name in model_names:
      click.echo(f'  {model_name}')


def run_with_errors(callback) -> int:
  try:
    callback()
  except GitCommandNotFound as exc:
    raise click.ClickException(f'git executable not found: {exc}') from exc
  except CliError as exc:
    raise click.ClickException(str(exc)) from exc
  return 0


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option(version=app_version(), prog_name='llml')
@click.option('--dry-run', is_flag=True, help='Print what would happen without making changes.')
@click.pass_context
def root(ctx: click.Context, dry_run: bool) -> None:
  inherited = ctx.obj or {}
  ctx.obj = State(settings=load_settings(), dry_run=dry_run or inherited.get('dry_run', False))


@root.command(add_help_option=False)
@click.option('-h', '--help', 'help_requested', is_flag=True, is_eager=True)
@click.argument('instance_name', required=False)
@click.argument('model_names', nargs=-1)
@click.pass_context
def sync(ctx: click.Context, help_requested: bool, instance_name: str | None, model_names: tuple[str, ...]) -> None:
  """Sync an instance's model files to disk."""
  state = state_from_context(ctx)

  def command() -> None:
    if help_requested or instance_name is None:
      if instance_name is None:
        print_instances_help('sync', state.settings)
        return
      instance = load_instance(state.settings, instance_name)
      print_models_help('sync', instance_name, instance)
      return

    instance = load_instance(state.settings, instance_name)
    for line in sync_models(instance, model_names, state.settings, state.dry_run):
      click.echo(line)

  run_with_errors(command)


@root.command(add_help_option=False)
@click.option('-h', '--help', 'help_requested', is_flag=True, is_eager=True)
@click.argument('instance_name', required=False)
@click.pass_context
def serve(ctx: click.Context, help_requested: bool, instance_name: str | None) -> None:
  """Serve an instance with llama-server."""
  state = state_from_context(ctx)

  def command() -> None:
    if help_requested or instance_name is None:
      if instance_name is None:
        print_instances_help('serve', state.settings)
        return
      instance = load_instance(state.settings, instance_name)
      print_models_help('serve', instance_name, instance)
      return

    instance = load_instance(state.settings, instance_name)
    code, line = serve_instance(instance_name, instance, state.settings, state.dry_run)
    if line:
      click.echo(line)
    if code:
      raise click.exceptions.Exit(code)

  run_with_errors(command)


@root.command(add_help_option=False)
@click.option('-h', '--help', 'help_requested', is_flag=True, is_eager=True)
@click.argument('instance_name', required=False)
@click.argument('model_names', nargs=-1)
@click.pass_context
def remove(ctx: click.Context, help_requested: bool, instance_name: str | None, model_names: tuple[str, ...]) -> None:
  """Delete an instance's model directories."""
  state = state_from_context(ctx)

  def command() -> None:
    if help_requested or instance_name is None:
      if instance_name is None:
        print_instances_help('remove', state.settings)
        return
      instance = load_instance(state.settings, instance_name)
      print_models_help('remove', instance_name, instance)
      return

    instance = load_instance(state.settings, instance_name)
    for line in remove_models(instance, model_names, state.settings, state.dry_run):
      click.echo(line)

  run_with_errors(command)


@root.command()
@click.pass_context
def tidy(ctx: click.Context) -> None:
  """Clean model content no instance defines anymore."""
  state = state_from_context(ctx)

  def command() -> None:
    lines = tidy_model_dir(state.settings, state.dry_run)
    if not lines:
      click.echo('model dir is already tidy')
      return
    for line in lines:
      click.echo(line)

  run_with_errors(command)


@root.command(name='list')
@click.argument('terms', nargs=-1)
@click.pass_context
def list_(ctx: click.Context, terms: tuple[str, ...]) -> None:
  """List installed instances and models, optionally filtered by terms."""
  state = state_from_context(ctx)
  run_with_errors(lambda: echo_matches(installed_instances(state.settings, terms)))


@root.command()
@click.argument('terms', nargs=-1)
@click.pass_context
def find(ctx: click.Context, terms: tuple[str, ...]) -> None:
  """Find instances and models defined in the config, installed or not."""
  state = state_from_context(ctx)
  run_with_errors(lambda: echo_matches(find_instances(state.settings, terms)))


@root.command()
@click.pass_context
def refresh(ctx: click.Context) -> None:
  """Refresh the config repo, cloning it if missing."""
  state = state_from_context(ctx)
  run_with_errors(lambda: click.echo(refresh_config_repo(state.settings, state.dry_run)))


@root.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
  """Show config and environment diagnostics."""
  state = state_from_context(ctx)

  def command() -> None:
    click.echo(f'llml {app_version()}')
    click.echo(f'config: {state.settings.user_config_path}')
    click.echo(f'config uri: {state.settings.config_uri}')
    click.echo(f'config dir: {state.settings.config_dir}')
    click.echo(f'model dir: {state.settings.model_dir}')
    click.echo(f'hf token: {"configured" if state.settings.hf_token else "not configured by llml"}')
    for package_name in ('click', 'huggingface-hub', 'GitPython'):
      try:
        click.echo(f'{package_name}: {version(package_name)}')
      except PackageNotFoundError:
        click.echo(f'{package_name}: missing')
    for name in ('git', 'llama-server'):
      path, found_version = executable_version(name)
      if path is None:
        click.echo(f'{name}: missing')
      else:
        click.echo(f'{name}: {path} ({found_version})')

  run_with_errors(command)


def main(argv: list[str] | None = None) -> int:
  args = list(sys.argv[1:] if argv is None else argv)
  dry_run = '--dry-run' in args
  args = [arg for arg in args if arg != '--dry-run']
  try:
    return root.main(args=args, prog_name='llml', standalone_mode=False, obj={'dry_run': dry_run})
  except click.ClickException as exc:
    exc.show()
    return exc.exit_code


if __name__ == '__main__':
  raise SystemExit(main())
