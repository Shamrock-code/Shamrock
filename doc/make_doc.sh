cd doxygen
for i in {1..3}
do
    doxygen dox.conf
    if [ $? -eq 0 ]
    then
        break
    fi
    echo "Doxygen failed, retrying ($i/3)"
done
if [ $i -eq 3 ]
then
    echo "Doxygen failed three times, giving up."
    exit 1
fi

set -e

cd ../mkdocs
cd docs/assets/figures
sh make_all_figs.sh
cd ../../..
mkdocs build
cd ..

rm -rf _build
mkdir _build
cd _build

mkdir doxygen
mkdir mkdocs

cp ../doxygen/warn_doxygen.txt doxygen
cp -r ../doxygen/html/* doxygen
cp -r ../mkdocs/site/* mkdocs

cp ../tmpindex.html index.html
