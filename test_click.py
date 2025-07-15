import click
@click.group()
def cli(): pass
@cli.command()
def run_daily(): click.echo('ran run daily')
if __name__=='__main__': cli()
