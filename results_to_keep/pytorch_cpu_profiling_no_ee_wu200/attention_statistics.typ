#align(center)[= Attention Layer Statistics]

Generated on 2025-03-18 21:56:59

Individual Layer Statistics

#figure(
  table(
    columns: 6,
    table.header([Layer], [Avg (ms)], [Std Dev], [Min (ms)], [Max (ms)], [Count]),
    [Layer_0], [14.455], [11.135], [11.148], [89.387], [50],
    [Layer_1], [17.717], [19.433], [11.218], [98.959], [50],
    [Layer_2], [13.388], [2.435], [11.196], [22.365], [50],
    [Layer_3], [18.133], [8.951], [14.597], [77.302], [50],
    [Layer_4], [16.513], [2.473], [14.223], [25.441], [50],
    [Layer_5], [16.120], [2.140], [13.944], [23.289], [50],
    [Layer_6], [16.285], [2.215], [14.644], [23.824], [50],
    [Layer_7], [15.805], [2.396], [13.932], [22.290], [50],
    [Layer_8], [15.853], [2.407], [13.755], [23.913], [50],
    [Layer_9], [15.512], [2.498], [13.735], [22.641], [50],
    [Layer_10], [15.265], [2.420], [13.614], [22.510], [50],
    [Layer_11], [13.546], [2.393], [11.860], [21.772], [50],
  ),
  caption: [Attention Latency by Individual Layer],
)

Layer Group Statistics

#figure(
  table(
    columns: 6,
    table.header([Layer], [Avg (ms)], [Std Dev], [Min (ms)], [Max (ms)], [Count]),
    [No early-exit], [14.777], [11.378], [11.148], [98.959], [200],
    [LPH], [16.763], [4.921], [13.944], [77.302], [200],
    [GAH], [15.609], [2.424], [13.614], [23.913], [200],
  ),
  caption: [Attention Latency by Layer Group],
)

Source: `attention_statistics.json`
