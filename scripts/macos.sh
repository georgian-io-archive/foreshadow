
echo Checking for Brew....

# Check if brew is installed and install it if not
command -v brew >/dev/null 2>&1 || { echo >&2 "Installing Homebrew Now"; \
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"; }

echo Installing pyenv

# Install pyenv

export HOMEBREW_NO_AUTO_UPDATE=1

brew update

if brew ls --versions pyenv > /dev/null; then
  echo pyenv is installed... skipping...
else
  brew install pyenv
fi


echo Installing neccesary python versions

pyenv install -s 3.5.5
pyenv install -s 3.6.5

echo Installing pyenv-virtualenv

if brew ls --versions pyenv-virtualenv > /dev/null; then
  echo pyenv-virutalenv is installed... skipping...
else
  brew install pyenv-virtualenv
  cat >> ~/.bash_profile <<EOF
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init -)"
  fi
  eval "$(pyenv virtualenv-init -)"
EOF
fi

source ~/.bash_profile

echo Setting up virtual environment

if [[ -z "${DEPLOY_ENV}" ]]; then
  pyenv local 3.5.5 3.6.5
  pyenv virtualenv venv
  pyenv local venv 3.6.5 3.5.5
else
  echo Already in virutalenv skipping setup...
fi

echo Installing requirements


if brew ls --versions swig > /dev/null; then
  echo swig is installed... skipping...
else
  brew install swig
fi

if brew ls --versions gcc@5 > /dev/null; then
  echo gcc-5 is installed... skipping...
else
  brew install gcc@5
fi

export CC=gcc-5
export CXX=g++-5

if [ ! -f requirements.txt ]; then
    echo "Could not find requirements will now exit"
    exit 2
fi

pip install -r pre_requirements.txt
pip install -r test_requirements.txt
pip install -r requirements.txt
