
# Step 1: for every model planning: [(bert, rtcp), text gen: [bart, llama2] we need to run the self simulation first.
# a pair (bert, bart), (bert, llama2), (rtcp, llama2)
# generating a new memory
sh scripts/run_self_simulation.sh

# Step 2: when step 1 is finished, run the run_mcts_online_eval
# pass the path to the new generated memory
sh scripts/run_mcts_online_eval.sh
