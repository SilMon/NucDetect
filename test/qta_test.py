import qtawesome as qta

# Get FontAwesome 5.x icons by name in various styles by name
from PyQt5 import QtWidgets

fa5_icon = qta.icon('fa5.flag')
fa5_button = QtWidgets.QPushButton(fa5_icon, 'Font Awesome! (regular)')

fa5s_icon = qta.icon('fa5s.flag')
fa5s_button = QtWidgets.QPushButton(fa5s_icon, 'Font Awesome! (solid)')

fa5b_icon = qta.icon('fa5b.github')
fa5b_button = QtWidgets.QPushButton(fa5b_icon, 'Font Awesome! (brands)')

# Get Elusive icons by name
asl_icon = qta.icon('ei.asl')
elusive_button = QtWidgets.QPushButton(asl_icon, 'Elusive Icons!')

# Get Material Design icons by name
apn_icon = qta.icon('mdi.access-point-network')
mdi_button = QtWidgets.QPushButton(apn_icon, 'Material Design Icons!')