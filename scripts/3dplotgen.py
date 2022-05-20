import plotly.graph_objects as go
import pickle
from plotly.offline import iplot

data = pickle.load(open("fulldata.plt", "rb"), encoding="bytes")
# Create 3D plotly fig where we add traces
fig = go.Figure()

for d in data:
    trace = go.Scatter3d(x=[], y=[], z=[], mode='markers')
    a,b,c = d
    for i,j,k in zip(a,b,c):
#         print(i,j,k)
#         break
        trace['x'] = list(trace['x']) + [i[0]]
        trace['y'] = list(trace['y']) + [j[0]]
        trace['z'] = list(trace['z']) + [k[0]]
    fig.add_trace(trace)

fig.update_traces(marker={'size': 5})
fig.show()
