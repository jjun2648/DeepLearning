from model import SimModel
import click



@click.command()
@click.option('-p', '--poke-name', type = click.STRING, default = 'abomasnow', help = '포켓몬의 이름')
@click.option('-m', '--model-name', type = click.STRING, default = 'vgg16', help = '모델의 이름')
def startmain(poke_name, model_name):
    
    SimModel(poke_name, model_name).showimage()
    


if __name__ == '__main__':
    startmain()