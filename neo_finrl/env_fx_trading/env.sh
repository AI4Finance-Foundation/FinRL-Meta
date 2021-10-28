DIR="./.venv"
python --version
pip --version
if [ -d "$DIR" ]; then
    echo "${DIR} is ready to use."
    source ./${DIR}/Scripts/activate
    echo "the venv actived, lanuch jupterlab for you, Sir..."
    python -m jupyterlab
else
    echo "Create virtual environment ${DIR}"
    python -m venv ${DIR}
    source ./${DIR}/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r ./neo_finrl/env_fx_trading/requirements.txt
    python -m jupyterlab
fi
