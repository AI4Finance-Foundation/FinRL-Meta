from ntpath import realpath
from wtpy.apps import WtBtAnalyst
from glob import glob
from tqdm.auto import tqdm
from os.path import dirname, basename
from click import command, option

# for i in tqdm(glob('./outputs_bt/*/funds.csv')):
#     folder = dirname(i)
#     name = folder[13:]
#     analyst = WtBtAnalyst()
#     folder = "./outputs_bt/%s/" % name
#     analyst.add_strategy(
#         name, folder=folder, init_capital=500000, rf=0.04, annual_trading_days=240)
#     try:
#         analyst.run_new('%s/%s_PnLAnalyzing.xlsx' % (folder, name))
#         # analyst.run('%s/%s_PnLAnalyzing.xlsx' % (folder, name))
#     except:
#         analyst.run('%s/%s_PnLAnalyzing.xlsx' % (folder, name))

@command()
@option('--path', '-p', 'path', default='./outputs_bt/*/')
def run(path):
    for i in tqdm(glob('%s/funds.csv'%path)):
        folder = dirname(i)
        name = basename(folder)
        analyst = WtBtAnalyst()
        analyst.add_strategy(
            name, folder='%s/'%folder, init_capital=500000, rf=0.04, annual_trading_days=240)
        try:
            analyst.run_new('%s/%s_PnLAnalyzing.xlsx' % (folder, name))
            # analyst.run('%s/%s_PnLAnalyzing.xlsx' % (folder, name))
        except:
            analyst.run('%s/%s_PnLAnalyzing.xlsx' % (folder, name))

if __name__ == '__main__':
    run()