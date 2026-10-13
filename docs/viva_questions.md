# Viva questions and short answers

**What problem does GeoMotion AI solve?**  
Noisy/sparse GPS does not sit on the true road. Map matching assigns observations to a connected path on a road network.

**Why not only snap to the nearest road?**  
Parallel roads: the closest segment at one sample can be the wrong corridor. A sequence model prefers a path that is also reachable.

**What is an HMM here?**  
Hidden states = candidate roads. Emissions = GPS points. Viterbi = dynamic programming for the most likely state sequence.

**Emission vs transition probability?**  
Emission: “how well does this road explain this GPS point?” (distance). Transition: “can the vehicle move from the previous road to this one given the observed displacement?”

**Did you use hmmlearn?**  
No. A small custom implementation is easier to explain and has no extra dependency.

**Is this machine learning?**  
Yes in the probabilistic-inference sense (HMM). No trained Random Forest/XGBoost, because we do not have per-point labels.

**Latitude/longitude?**  
Not established. Values overlap a projected graph. Euclidean distance in that plane.

**How do you know reference IDs are nodes?**  
They all exist in `vertexMap.txt`. `adjacentList` has no edge id column. Consecutive IDs usually form edges.

**How are sparse and dense files aligned?**  
By row order (`order_index`), not by the ID column.

**What would you add next?**  
Known CRS, denser point labels, parameter search on held-out trips, OSM polyline geometry when node sequences match.
