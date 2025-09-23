source set.sh
mkdir ./DATA
mkdir ./results
mkdir ./results/recall@20
mkdir ./results/recall@100
mkdir ./results/time-log
mkdir ./results/space-log

rm -rf build
mkdir build
cd build

cmake ..
make clean
make -j 40

cd ..

mkdir ./figure

for dataset in "${datasets[@]}";
do
  echo $dataset
  mkdir ./DATA/${dataset}
  mkdir ./results/recall@20/${dataset}
  mkdir ./results/recall@100/${dataset}
  mkdir ./results/time-log/${dataset}
  mkdir ./results/space-log/${dataset}
  mkdir ./figure/${dataset}
done