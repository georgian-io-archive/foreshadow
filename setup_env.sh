unamestr=$(uname)
if [[ "$unamestr" == 'Linux' ]]; then
   echo Automatic install on this platform is not supported yet. Please read the developers guide to get an environment setup
elif [[ "$unamestr" == 'Darwin' ]]; then
   bash scripts/macos.sh
fi

