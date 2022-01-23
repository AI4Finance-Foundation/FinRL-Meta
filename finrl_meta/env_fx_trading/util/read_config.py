import json


class EnvConfig():
    """environment configuration from json file
       tgym requires you configure your own parameters in json file.
        Args:
            config_file path/file.json

    """

    def __init__(self, config_file='./neo_finrl/env_fx_trading/config/gdbusd-test-1.json'):
        self.config = {}
        with open(config_file) as j:
            self.config = json.load(j)

    def env_parameters(self, item=''):
        """environment variables
        """
        return self.config["env"][item] if item else self.config["env"]

    def symbol(self, asset="GBPUSD", item=''):
        """get trading pair (symbol) information

        Args:
            asset (str, optional): symbol in config. Defaults to "GBPUSD".
            item (str, optional): name of item, if '' return dict, else return item value. Defaults to ''.

        Returns:
            [type]: [description]
        """
        if item:
            return self.config["symbol"][asset][item]
        else:
            return self.config["symbol"][asset]

    def trading_hour(self, place="New York"):
        """forex trading hour from different markets

        Args:
            place (str, optional): [Sydney,Tokyo,London] Defaults to "New York".

        Returns:
            [dict]: from time, to time
        """
        if place:
            return self.config["trading_hour"][place]
        else:
            return self.config["trading_hour"]


if __name__ == '__main__':
    cf = EnvConfig()
    print(f'{cf.env_parameters()}')
    print(cf.env_parameters("observation_list"))
    print(f'asset_col: {cf.env_parameters()["asset_col"]}')
    print(cf.symbol(asset="GBPUSD")["point"])
    print(f'trading hour new york: {cf.trading_hour("new york")}')
