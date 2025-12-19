# Initialise dataframes for ground and header test sets
X_test_grnd = pd.DataFrame(columns = ['xpos','cpos','distance','angle','header_tag'], dtype='float64')
X_test_head = pd.DataFrame(columns = ['xpos','cpos','distance','angle','header_tag'], dtype='float64')
i = 0

# Create array of shots
for x_pos in range(0,int(PITCH_LENGTH_X/2 + 1)):
    for y_pos in range(0, int(PITCH_WIDTH_Y + 1)):
        c_pos = y_pos - PITCH_WIDTH_Y/2
        angle_denominator = (x_pos**2 + c_pos**2 - (GOAL_WIDTH_Y/2)**2)
        if angle_denominator == 0:
            angle = np.pi/2
        else:
            angle = np.arctan(2*(GOAL_WIDTH_Y/2)*x_pos/angle_denominator)
        if angle < 0:
            angle = np.pi + angle
        distance = np.sqrt(x_pos**2 + c_pos**2)
        X_test_grnd.loc[i,:] = [x_pos, c_pos, distance, angle, 0]
        X_test_head.loc[i,:] = [x_pos, c_pos, distance, angle, 1]
        i += 1

prob_goal_grnd = neural_net.predict([X_test_grnd])[:,1].reshape(int(1+PITCH_LENGTH_X/2),int(1+PITCH_WIDTH_Y))
prob_goal_head = neural_net.predict([X_test_head])[:,1].reshape(int(1+PITCH_LENGTH_X/2),int(1+PITCH_WIDTH_Y))

# %% Plot xG model

# Overwrite rcParams 
mpl.rcParams['xtick.color'] = "white"
mpl.rcParams['ytick.color'] = "white"
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

# Plot pitches
pitch = VerticalPitch(half=True,pitch_color='#313332', line_color='white', linewidth=1, stripe=False)
fig, ax = pitch.grid(nrows=1, ncols=2, grid_height=0.75, space = 0.1, axis=False)
fig.set_size_inches(10, 5.5)
fig.set_facecolor('#313332')

# Add xG maps and contours
pos1 = ax['pitch'][0].imshow(prob_goal_grnd, extent = (80,0,60,120) ,aspect='equal',vmin=-0.04,vmax=0.4,cmap=plt.cm.inferno)
pos2 = ax['pitch'][1].imshow(prob_goal_head, extent = (80,0,60,120) ,aspect='equal',vmin=-0.04,vmax=0.4,cmap=plt.cm.inferno)
cs1 = ax['pitch'][0].contour(prob_goal_grnd, extent = (1,80,120,60), levels = [0.01,0.05,0.2,0.5], colors = ['darkgrey','darkgrey','darkgrey','k'], linestyles = 'dotted')
cs2 = ax['pitch'][1].contour(prob_goal_head, extent = (1,80,120,60), levels = [0.01,0.05,0.2,0.5], colors = ['darkgrey','darkgrey','darkgrey','k'], linestyles = 'dotted')
ax['pitch'][0].clabel(cs1)
ax['pitch'][1].clabel(cs2)

# Title
fig.text(0.045,0.9,"Expected Goals - Neural Network", fontsize=16, color="white", fontweight="bold")
fig.text(0.045,0.85,"Trained on all 40,000+ shots during the 2017/18 season across Europe's 'big five' Leagues", fontsize=14, color="white", fontweight="regular")
fig.text(0.12,0.76,"Shot Type: Left or Right Foot", fontsize=12, color="white", fontweight="bold")
fig.text(0.66,0.76,"Shot Type: Header", fontsize=12, color="white", fontweight="bold")

# Colourbar
cbar = fig.colorbar(pos2, ax=ax['pitch'][1], location="bottom",  fraction = 0.04, pad = 0.0335)
cbar.ax.set_ylabel('xG', loc="bottom", color = "white", fontweight="bold", rotation=0, labelpad=20)

# Footer text
fig.text(0.255, 0.09, "Created by Jake Kolliari. Data provided by Wyscout.com",
         fontstyle="italic", ha="center", fontsize=9, color="white")  

# Format and show
plt.tight_layout()
plt.show()