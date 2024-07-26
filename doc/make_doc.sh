cd doxygen
doxygen dox.conf

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