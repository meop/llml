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
from llml.instances import (
  find_instances,
  installed_instances,
  load_instance,
  pinpoint_installed_model,
  pinpoint_instance,
  wide_models,
)
from llml.settings import Settings, load_settings


@dataclass
class State:
  settings: Settings
  dry_run: bool


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
@click.option('-n', '--dry-run', is_flag=True, help='Print what would happen without making changes.')
@click.pass_context
def root(ctx: click.Context, dry_run: bool) -> None:
  inherited = ctx.obj or {}
  ctx.obj = State(settings=load_settings(), dry_run=dry_run or inherited.get('dry_run', False))


@root.command()
@click.argument('terms', nargs=-1)
@click.pass_context
def sync(ctx: click.Context, terms: tuple[str, ...]) -> None:
  """Sync (download) files for every model matching the terms (all if none)."""
  state = state_from_context(ctx)

  def command() -> None:
    pairs = wide_models(state.settings, terms)
    if not pairs:
      click.echo('no models match')
      return
    for instance_name in dict.fromkeys(name for name, _ in pairs):
      instance = load_instance(state.settings, instance_name)
      models = tuple(model for name, model in pairs if name == instance_name)
      for line in sync_models(instance, models, state.settings, state.dry_run):
        click.echo(line)

  run_with_errors(command)


@root.command()
@click.argument('terms', nargs=-1)
@click.pass_context
def remove(ctx: click.Context, terms: tuple[str, ...]) -> None:
  """Remove one installed model, pinpointed from the terms."""
  state = state_from_context(ctx)

  def command() -> None:
    if not terms:
      raise CliError('remove needs at least one term; run `list` to see installed models')
    instance_name, model_name = pinpoint_installed_model(state.settings, terms)
    instance = load_instance(state.settings, instance_name)
    for line in remove_models(instance, (model_name,), state.settings, state.dry_run):
      click.echo(line)

  run_with_errors(command)


@root.command()
@click.argument('terms', nargs=-1)
@click.pass_context
def serve(ctx: click.Context, terms: tuple[str, ...]) -> None:
  """Serve one instance, pinpointed from the terms, with llama-server."""
  state = state_from_context(ctx)

  def command() -> None:
    if not terms:
      raise CliError('serve needs at least one term; run `find` to see instances')
    instance_name = pinpoint_instance(state.settings, terms)
    instance = load_instance(state.settings, instance_name)
    code, line = serve_instance(instance_name, instance, state.settings, state.dry_run)
    if line:
      click.echo(line)
    if code:
      raise click.exceptions.Exit(code)

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
  dry_run_flags = {'-n', '--dry-run'}
  dry_run = any(arg in dry_run_flags for arg in args)
  args = [arg for arg in args if arg not in dry_run_flags]
  try:
    return root.main(args=args, prog_name='llml', standalone_mode=False, obj={'dry_run': dry_run})
  except click.ClickException as exc:
    exc.show()
    return exc.exit_code


if __name__ == '__main__':
  raise SystemExit(main())
