grep -E "iter [0-9]*0" experiments/$1/log.txt | grep -Eo "smoothed loss [0-9]+\.[0-9]+"  | grep -Eo "[0-9]+\.[0-9]+" > experiments/$1/train_loss_every_10.txt

# create dev loss file
grep  "dev loss:" experiments/$1/log.txt | grep -Eo "[0-9]+\.[0-9]+" > experiments/$1/dev_loss_tmp.txt
grep  "dev loss:" experiments/$1/log.txt | grep -Eo "Iter [0-9]+" | grep -Eo "[0-9]+" > experiments/$1/eval_its.txt
paste experiments/$1/eval_its.txt experiments/$1/dev_loss_tmp.txt  > experiments/$1/dev_loss.txt

# create f1 score file
grep -Eo "Dev F1 score: [0-9]+\.[0-9]+" experiments/$1/log.txt | grep -Eo "[0-9]+\.[0-9]+" > experiments/$1/F1_loss_tmp.txt
paste experiments/$1/eval_its.txt experiments/$1/F1_loss_tmp.txt  > experiments/$1/F1_loss.txt

grep -Eo "Dev EM score: [0-9]+\.[0-9]+" experiments/$1/log.txt | grep -Eo "[0-9]+\.[0-9]+" > experiments/$1/EM_loss_tmp.txt
paste experiments/$1/eval_its.txt experiments/$1/EM_loss_tmp.txt  > experiments/$1/EM_loss.txt

rm experiments/$1/F1_loss_tmp.txt
rm experiments/$1/EM_loss_tmp.txt
rm experiments/$1/dev_loss_tmp.txt