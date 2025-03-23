#align(center)[= Attention Layer Statistics]

Generated on 2025-03-18 22:26:34

Individual Layer Statistics

#figure(
  table(
    columns: 6,
    table.header([Layer], [Avg (ms)], [Std Dev], [Min (ms)], [Max (ms)], [Count]),
    [Layer_0], [2.880], [9.757], [1.284], [70.473], [50],
    [Layer_1], [5.588], [17.110], [1.134], [72.840], [50],
    [Layer_2], [1.363], [0.242], [1.119], [2.165], [50],
    [Layer_3], [5.258], [2.587], [4.684], [23.045], [50],
    [Layer_4], [4.911], [0.387], [4.661], [6.328], [50],
    [Layer_5], [5.074], [0.314], [4.656], [5.996], [50],
    [Layer_6], [4.865], [0.280], [4.625], [6.075], [50],
    [Layer_7], [4.652], [0.278], [4.439], [5.717], [50],
    [Layer_8], [4.625], [0.212], [4.491], [5.911], [50],
    [Layer_9], [4.639], [0.310], [4.425], [6.012], [50],
    [Layer_10], [4.588], [0.164], [4.487], [5.501], [50],
    [Layer_11], [1.267], [0.204], [1.092], [2.147], [50],
  ),
  caption: [Attention Latency by Individual Layer],
)

Layer Group Statistics

#figure(
  table(
    columns: 6,
    table.header([Layer Group], [Avg (ms)], [Std Dev], [Min (ms)], [Max (ms)], [Count]),
    [GAH], [4.626], [0.247], [4.425], [6.012], [200],
    [LPH], [5.027], [1.324], [4.625], [23.045], [200],
    [No early-exit], [2.775], [9.931], [1.092], [72.840], [200],
  ),
  caption: [Attention Latency by Layer Group],
)

Source: `attention_statistics.json`
