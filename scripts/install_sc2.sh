#!/bin/bash
# Install SC2 and add the custom maps

echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        unzip -P iagreetotheeula SC2.4.10.zip -d /
        rm -rf SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

cd ..
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip

echo 'StarCraft II and SMAC are installed.'

