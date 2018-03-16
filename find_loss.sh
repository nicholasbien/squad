echo "##############################################"
echo "Statistics for experiment: " $1
echo "##############################################"

echo "Train Stats"
grep  "Train F1" experiments/$1/log.txt #| grep -Eo "Train F1 score: [0-9]+([.][0-9]+)?"

echo "Dev Stats"
grep  "Dev F1" experiments/$1/log.txt #| grep -Eo "Dev F1 score: [0-9]+([.][0-9]+)?"

echo "Dev Stats"
grep  "dev loss" experiments/$1/log.txt #| grep -Eo "Dev F1 score: [0-9]+([.][0-9]+)?"
