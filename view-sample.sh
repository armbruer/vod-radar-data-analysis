#!/bin/sh

cd ./output/analysis/sem
cur=$(pwd)

id=$1
xdg-open "${cur}/sem-extremum-highlighted-${id}.png"
xdg-open "${cur}/sem-extremum-highlighted-${id}.html"
xdg-open "${cur}/sem-radar-${id}.png"
xdg-open "${cur}/sem-radar-${id}.html"
cd -

