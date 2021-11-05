DIR="./.venv"
python --version
pip --version
if [ -d "$DIR" ]; then
    echo "${DIR} is ready to use."
    source ./${DIR}/Scripts/activate
    echo "the venv actived"

else
    echo "Create virtual environment ${DIR}"
    python -m venv ${DIR}
    source ./${DIR}/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r ./neo_finrl/env_fx_trading/requirements.txt
fi

if [ $1 = "j" ]; then
    echo "start jupyter notebook, Sir" 
    python -m jupyterlab
else
    echo "start virtual python env shell, Sir"
fi
