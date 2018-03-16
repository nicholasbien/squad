
for var in "$@"
do
    source gather_results.sh $var
done

python plot_graph.py --experiment_names "$@"