import sys
import os
import numpy as np
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QLabel, QPushButton, QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QLineEdit, QTabWidget
from PySide2.QtCore import Signal, Slot, QFile, QObject, Qt
from PySide2.QtGui import QPixmap
sys.path.insert(0, os.path.join('..', 'apsimulator'))
import galaxy_generation
import math_utils
import parallel_utils
import image_utils
import plot_utils


class MainWindow(QObject):

    def __init__(self, parent=None):

        super(MainWindow, self).__init__(parent)

        self.initialize_galaxy()

        self.load_ui()

        self.disable_auto_generate = False
        self.disable_parameter_updates = False

        self.setup_galaxy_frame()
        self.setup_generation_control()
        self.setup_orientation_adjustment()
        self.setup_morphology_adjustment()
        self.setup_disk_component_adjustment()
        self.setup_bulge_component_adjustment()
        self.setup_visualization_control()

        parallel_utils.set_number_of_threads(12)

    # ****** UI loading ******

    def load_ui(self):
        ui_file = QFile('galaxy_editor.ui')
        ui_file.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        self.window.show()

    # ****** Galaxy generation ******

    def initialize_galaxy(self):
        self.galaxy = galaxy_generation.Galaxy()

    def setup_galaxy_frame(self):
        self.galaxy_frame_label = self.window.findChild(QLabel, 'galaxyFrameLabel')

    def setup_generation_control(self):
        self.setup_auto_generate_checkbox()
        self.setup_resolution_combobox()
        self.setup_scale_adjustment()
        self.setup_generate_button()

    def setup_auto_generate_checkbox(self):
        self.auto_generate_checkbox = self.window.findChild(QCheckBox, 'autoGenerateCheckBox')

    def auto_generate_galaxy_image(self, *args):
        if self.auto_generate_checkbox.isChecked() and not self.disable_auto_generate:
            self.generate_galaxy_image()

    def setup_resolution_combobox(self):
        self.resolution_combo_box = self.window.findChild(QComboBox, 'resolutionComboBox')
        resolutions = galaxy_generation.Galaxy.GUI_param_ranges['resolution']
        texts = [u'{:d}\u00B3'.format(resolution) for resolution in resolutions]
        self.resolution_combo_box.insertItems(0, texts)

        @Slot(int)
        def resolution_combo_box_action(index):
            if not self.disable_parameter_updates:
                self.galaxy.set_resolution(resolutions[index])
            self.auto_generate_galaxy_image()

        self.resolution_combo_box.setCurrentIndex(resolutions.index(self.galaxy.get_resolution()))
        self.resolution_combo_box.currentIndexChanged.connect(resolution_combo_box_action)

    def setup_scale_adjustment(self):
        self.scale_slider = self.window.findChild(QSlider, 'scaleSlider')
        self.scale_spinbox = self.window.findChild(QDoubleSpinBox, 'scaleSpinBox')

        def scale_setter(scale):
            self.galaxy.set_scale(scale)
            self.auto_generate_galaxy_image()

        setup_slider_and_spinbox(self.scale_slider, self.scale_spinbox, *galaxy_generation.Galaxy.GUI_param_ranges['scale'], scale_setter)

        self.scale_spinbox.setValue(self.galaxy.get_scale())

    def setup_generate_button(self):
        self.generate_button = self.window.findChild(QPushButton, 'generateButton')

        @Slot()
        def generate_button_action():
            self.generate_galaxy_image()

        self.generate_button.clicked.connect(generate_button_action)

    def generate_galaxy_image(self):
        image_path = os.path.join(os.getcwd(), 'galaxy.png')
        image_values = self.convert_intensity_to_image_values(self.galaxy.compute_intensity())
        plot_utils.save_pure_image(image_path, image_values)
        self.galaxy_frame_label.setPixmap(QPixmap(image_path).scaled(self.galaxy_frame_label.width(),
                                                                     self.galaxy_frame_label.height(),
                                                                     aspectMode=Qt.KeepAspectRatio,
                                                                     mode=Qt.FastTransformation))

    # ****** Orientation adjustment ******

    def setup_orientation_adjustment(self):
        self.setup_polar_angle_adjustment()
        self.setup_azimuth_angle_adjustment()

    def setup_polar_angle_adjustment(self):
        self.polar_angle_slider = self.window.findChild(QSlider, 'polarAngleSlider')
        self.polar_angle_spinbox = self.window.findChild(QDoubleSpinBox, 'polarAngleSpinBox')

        def polar_angle_setter(angle):
            if not self.disable_parameter_updates:
                self.galaxy.get_orientation().set_polar_angle(math_utils.radian_from_degree(angle))
            self.auto_generate_galaxy_image()

        setup_slider_and_spinbox(self.polar_angle_slider, self.polar_angle_spinbox, 0, 180, 1, polar_angle_setter)

        self.disable_parameter_updates = True
        self.polar_angle_spinbox.setValue(math_utils.degree_from_radian(self.galaxy.get_orientation().get_polar_angle()))
        self.disable_parameter_updates = False

    def setup_azimuth_angle_adjustment(self):
        self.azimuth_angle_slider = self.window.findChild(QSlider, 'azimuthAngleSlider')
        self.azimuth_angle_spinbox = self.window.findChild(QDoubleSpinBox, 'azimuthAngleSpinBox')

        def azimuth_angle_setter(angle):
            if not self.disable_parameter_updates:
                self.galaxy.get_orientation().set_azimuth_angle(math_utils.radian_from_degree(angle))
            self.auto_generate_galaxy_image()

        setup_slider_and_spinbox(self.azimuth_angle_slider, self.azimuth_angle_spinbox, 0, 360, 1, azimuth_angle_setter)

        self.disable_parameter_updates = True
        self.azimuth_angle_spinbox.setValue(math_utils.degree_from_radian(self.galaxy.get_orientation().get_azimuth_angle()))
        self.disable_parameter_updates = False

    # ****** Morphology adjustment ******

    def setup_morphology_adjustment(self):
        self.morphology_widget_value_setters = {}
        self.setup_winding_number_adjustment()
        self.setup_bulge_to_arm_ratio_adjustment()
        self.setup_arm_scale_adjustment()
        self.setup_arm_orientations_adjustment()
        self.setup_reset_morphology_parameters_button()

    def setup_morphology_slider_spinbox_adjustment(self, quantity_name, widget_name_base):

        slider = self.window.findChild(QSlider, '{}Slider'.format(widget_name_base))
        spinbox = self.window.findChild(QDoubleSpinBox, '{}SpinBox'.format(widget_name_base))

        self.morphology_widget_value_setters[quantity_name] = lambda value: spinbox.setValue(value)

        setup_slider_and_spinbox(slider, spinbox,
                                 *galaxy_generation.GalaxyMorphology.GUI_param_ranges[quantity_name],
                                 lambda value: self.morphology_parameter_setter_template(quantity_name, value))

        self.disable_parameter_updates = True
        spinbox.setValue(self.galaxy.get_morphology().get(quantity_name))
        self.disable_parameter_updates = False

    def morphology_parameter_setter_template(self, quantity_name, value):
        if not self.disable_parameter_updates:
            self.galaxy.get_morphology().set(quantity_name, value)
        self.auto_generate_galaxy_image()

    def setup_winding_number_adjustment(self):
        self.setup_morphology_slider_spinbox_adjustment('winding_number', 'windingNumber')

    def setup_bulge_to_arm_ratio_adjustment(self):
        self.setup_morphology_slider_spinbox_adjustment('bulge_to_arm_ratio', 'bulgeToArmRatio')

    def setup_arm_scale_adjustment(self):
        self.setup_morphology_slider_spinbox_adjustment('arm_scale', 'armScale')

    def setup_arm_orientations_adjustment(self):
        self.arms_combobox = self.window.findChild(QComboBox, 'armsComboBox')
        self.new_arm_button = self.window.findChild(QPushButton, 'newArmButton')
        self.remove_arm_button = self.window.findChild(QPushButton, 'removeArmButton')
        self.arm_angle_label = self.window.findChild(QLabel, 'armAngleLabel')
        self.arm_angle_slider = self.window.findChild(QSlider, 'armAngleSlider')
        self.arm_angle_spinbox = self.window.findChild(QDoubleSpinBox, 'armAngleSpinBox')

        @Slot(int)
        def arms_combobox_action(index):
            self.arm_angle_label.setText(self.arms_combobox.itemText(index))
            self.disable_auto_generate = True # Make sure not to recompute the image when updating the arm angle slider
            self.disable_parameter_updates = True # Make sure not to update the morphology angle attribute, as this would be redundant
            self.arm_angle_spinbox.setValue(math_utils.degree_from_radian(self.galaxy.get_morphology().get_arm_orientation_angle(index)))
            self.disable_parameter_updates = False
            self.disable_auto_generate = False

        @Slot()
        def new_arm_button_action():
            new_index = self.galaxy.get_morphology().get_number_of_arms()
            self.galaxy.get_morphology().add_arm(0)
            self.arms_combobox.addItem('Arm {:d}'.format(new_index+1))
            self.arms_combobox.setCurrentIndex(new_index)
            self.auto_generate_galaxy_image()

        @Slot()
        def remove_arm_button_action():
            if self.galaxy.get_morphology().get_number_of_arms() < 2: # Do nothing if only one arm is left
                return
            current_index = self.arms_combobox.currentIndex()
            self.galaxy.get_morphology().remove_arm(current_index)
            self.arms_combobox.removeItem(current_index)
            for index in range(current_index, self.galaxy.get_morphology().get_number_of_arms()):
                self.arms_combobox.setItemText(index, 'Arm {:d}'.format(index+1)) # Update remaining arm numbers
            self.arm_angle_label.setText(self.arms_combobox.itemText(self.arms_combobox.currentIndex()))
            self.auto_generate_galaxy_image()

        self.arms_combobox.currentIndexChanged.connect(arms_combobox_action)
        self.new_arm_button.clicked.connect(new_arm_button_action)
        self.remove_arm_button.clicked.connect(remove_arm_button_action)

        def arm_orientation_setter(arm_angle):
            if not self.disable_parameter_updates:
                self.galaxy.get_morphology().set_arm_orientation(self.arms_combobox.currentIndex(), math_utils.radian_from_degree(arm_angle))
            self.auto_generate_galaxy_image()

        setup_slider_and_spinbox(self.arm_angle_slider, self.arm_angle_spinbox, 0, 360, 1, arm_orientation_setter)

        arms = ['Arm {:d}'.format(index+1) for index in range(self.galaxy.get_morphology().get_number_of_arms())]
        self.arms_combobox.insertItems(0, arms)

    def setup_reset_morphology_parameters_button(self):

        self.reset_morphology_parameters_button = self.window.findChild(QPushButton, 'resetMorphologyParametersButton')

        @Slot()
        def reset_morphology_parameters_button_action():
            morphology = self.galaxy.get_morphology()
            default_params = galaxy_generation.GalaxyMorphology.get_default_params()

            self.disable_auto_generate = True # Make sure not to recompute the image for each widget
            for name in ['winding_number', 'bulge_to_arm_ratio', 'arm_scale']:
                self.morphology_widget_value_setters[name](default_params[name]) # Set widget values to default, which will also update morphology attributes
            self.disable_auto_generate = False

            morphology.set_arm_orientations(default_params['arm_orientations'])
            arms = ['Arm {:d}'.format(index+1) for index in range(morphology.get_number_of_arms())]
            self.arms_combobox.clear()
            self.arms_combobox.insertItems(0, arms)

            self.auto_generate_galaxy_image() # Now we can recompute the image

        self.reset_morphology_parameters_button.clicked.connect(reset_morphology_parameters_button_action)

    # ****** Disk component adjustment ******

    def setup_disk_component_adjustment(self):
        self.disk_component_parameter_tab_widget = self.window.findChild(QTabWidget, 'diskComponentParameterTabWidget')
        self.setup_disk_component_parameter_controls()
        self.setup_disk_component_selection()

    def setup_disk_component_selection(self):
        self.disk_component_types = galaxy_generation.GalaxyDiskComponent.get_component_types()
        self.disk_component_labels = {component_type: [] for component_type in self.disk_component_types}
        self.setup_reset_disk_component_parameters_button()
        self.setup_disk_components_combobox()
        self.setup_disk_component_type_combobox()
        self.setup_new_disk_component_button()
        self.setup_remove_disk_component_button()
        self.setup_disk_component_name_control()

    def setup_disk_component_parameter_controls(self):
        self.disk_component_parameter_control_tabs = {}
        self.disk_component_parameter_control_widgets = {}
        self.disk_component_widget_value_setters = {}
        self.setup_disk_component_active_state_adjustment()
        self.setup_disk_component_emissive_state_adjustment()
        self.setup_disk_component_strength_scale_adjustment()
        self.setup_disk_component_disk_extent_adjustment()
        self.setup_disk_component_disk_thickness_adjustment()
        self.setup_disk_component_arm_narrowness_adjustment()
        self.setup_disk_component_twirl_factor_adjustment()
        self.setup_disk_component_seed_adjustment()
        self.setup_disk_component_number_of_octaves_adjustment()
        self.setup_disk_component_initial_frequency_adjustment()
        self.setup_disk_component_lacunarity_adjustment()
        self.setup_disk_component_persistence_adjustment()
        self.setup_disk_component_noise_threshold_adjustment()
        self.setup_disk_component_noise_cutoff_adjustment()
        self.setup_disk_component_noise_exponent_adjustment()
        self.setup_disk_component_noise_offset_adjustment()

    def set_disk_component_parameter_control_widget_enabled_states(self, enabled):
        for widget_list in self.disk_component_parameter_control_widgets.values():
            for widget in widget_list:
                widget.setEnabled(enabled)

    def update_disk_component_parameter_control_widget_values(self, params, excluded_param_names=[], only_active_tab=False, update_disk_component_parameters=True):
        auto_generate_was_disabled = self.disable_auto_generate
        self.disable_auto_generate = True # Make sure not to recompute the image for each widget

        self.disable_parameter_updates = not update_disk_component_parameters # We might not want to update the component class attributes automatically

        active_tab = self.disk_component_parameter_tab_widget.currentWidget().objectName()
        for name in self.disk_component_widget_value_setters:
            if name not in excluded_param_names and not (only_active_tab and self.disk_component_parameter_control_tabs[name] != active_tab):
                self.disk_component_widget_value_setters[name](params[name])

        self.disable_parameter_updates = False

        self.disable_auto_generate = auto_generate_was_disabled

    def setup_disk_component_type_combobox(self):

        self.disk_component_type_combobox = self.window.findChild(QComboBox, 'diskComponentTypeComboBox')

        @Slot(str)
        def disk_component_type_combobox_action(text):
            # Update the contents of the disk component combo box
            self.disk_components_combobox.clear()
            self.disk_components_combobox.insertItems(0, self.disk_component_labels[text])

        self.disk_component_type_combobox.currentIndexChanged[str].connect(disk_component_type_combobox_action)

        self.disk_component_type_combobox.insertItems(0, list(self.disk_component_types.keys()))

    def setup_disk_components_combobox(self):

        self.disk_components_combobox = self.window.findChild(QComboBox, 'diskComponentsComboBox')

        @Slot(int)
        def disk_components_combobox_action(index):
            # Update the name text and values of the adjustment controls
            component_label = self.disk_components_combobox.currentText()
            self.disk_component_name_line_edit.setText(component_label)
            if len(component_label.strip()) == 0: # Simply disable the relevant widgets if there is no component
                self.set_disk_component_parameter_control_widget_enabled_states(False)
            else:
                self.set_disk_component_parameter_control_widget_enabled_states(True) # Make sure control widgets are enabled
                disk_component = self.galaxy.get_disk_component(component_label)
                # Update widget values, but make sure the setters don't update the component class attributes, which would be redundant
                self.update_disk_component_parameter_control_widget_values(disk_component.get_params(), update_disk_component_parameters=False)

        self.disk_components_combobox.currentIndexChanged.connect(disk_components_combobox_action)
        self.set_disk_component_parameter_control_widget_enabled_states(False) # Controls are disabled initially

    def setup_new_disk_component_button(self):

        self.new_disk_component_button = self.window.findChild(QPushButton, 'newDiskComponentButton')

        @Slot()
        def new_disk_component_button_action():
            # Create a new disk component with a unique label
            component_type_name = self.disk_component_type_combobox.currentText()

            all_component_labels = [item for sublist in self.disk_component_labels.values() for item in sublist]
            component_label = component_type_name + '_1'
            while component_label in all_component_labels: # Keep incrementing the index until the label is unique
                component_label = component_label[:-1] + str(int(component_label[-1]) + 1)

            self.disk_component_labels[component_type_name].append(component_label)
            self.galaxy.add_disk_component(self.disk_component_types[component_type_name](component_label))
            self.disk_components_combobox.addItem(component_label)
            self.disk_components_combobox.setCurrentIndex(len(self.disk_component_labels[component_type_name])-1)
            self.auto_generate_galaxy_image()

        self.new_disk_component_button.clicked.connect(new_disk_component_button_action)

    def setup_remove_disk_component_button(self):

        self.remove_disk_component_button = self.window.findChild(QPushButton, 'removeDiskComponentButton')

        @Slot()
        def remove_disk_component_button_action():
            # Remove the currently selected disk component
            component_type_name = self.disk_component_type_combobox.currentText()
            if len(self.disk_component_labels[component_type_name]) == 0: # Nothing to remove if no components are shown
                return
            component_label = self.disk_components_combobox.currentText()
            self.disk_component_labels[component_type_name].remove(component_label)
            self.galaxy.remove_disk_component(component_label)
            self.disk_components_combobox.removeItem(self.disk_components_combobox.currentIndex())
            self.auto_generate_galaxy_image()

        self.remove_disk_component_button.clicked.connect(remove_disk_component_button_action)

    def setup_disk_component_name_control(self):

        self.disk_component_name_line_edit = self.window.findChild(QLineEdit, 'diskComponentNameLineEdit')
        self.set_disk_component_name_button = self.window.findChild(QPushButton, 'setDiskComponentNameButton')

        @Slot()
        def set_disk_component_name_button_action():
            # Update the current component label to the content of the line edit
            component_type_name = self.disk_component_type_combobox.currentText()
            current_component_label = self.disk_components_combobox.currentText()
            new_component_label = self.disk_component_name_line_edit.text()
            if len(new_component_label.strip()) == 0: # Skip updating if the line edit is empty
                return
            all_component_labels = [item for sublist in self.disk_component_labels.values() for item in sublist]
            if new_component_label in all_component_labels: # Skip updating if the new label already exists
                return
            component_index = self.disk_component_labels[component_type_name].index(current_component_label)
            self.disk_component_labels[component_type_name][component_index] = new_component_label
            self.galaxy.set_disk_component_label(current_component_label, new_component_label)
            self.disk_components_combobox.setItemText(self.disk_components_combobox.currentIndex(), new_component_label)

        self.set_disk_component_name_button.clicked.connect(set_disk_component_name_button_action)

    def setup_reset_disk_component_parameters_button(self):

        self.reset_disk_component_parameters_button = self.window.findChild(QPushButton, 'resetDiskComponentParametersButton')

        self.disk_component_parameter_control_widgets['reset'] = [self.reset_disk_component_parameters_button]

        @Slot()
        def reset_disk_component_parameters_button_action():
            # Set the parameters of the currently selected disk component to the type's default parameters
            component_label = self.disk_components_combobox.currentText()
            if len(component_label.strip()) == 0: # Skip updating if the line edit is empty
                return
            disk_component = self.galaxy.get_disk_component(component_label)
            default_params = disk_component.__class__.get_default_params()
            # Set widget values to default in the current tab, which will also update the component class attributes
            self.update_disk_component_parameter_control_widget_values(default_params, only_active_tab=True, excluded_param_names=['active', 'seed'])
            self.auto_generate_galaxy_image()

        self.reset_disk_component_parameters_button.clicked.connect(reset_disk_component_parameters_button_action)

    def setup_disk_component_active_state_adjustment(self):
        self.disk_component_active_checkbox = self.window.findChild(QCheckBox, 'diskComponentActiveCheckBox')

        quantity_name = 'active'
        self.disk_component_parameter_control_widgets[quantity_name] = [self.disk_component_active_checkbox]
        self.disk_component_widget_value_setters[quantity_name] = lambda active: self.disk_component_active_checkbox.setChecked(active)

        @Slot(int)
        def disk_component_active_checkbox_action(state):
            if not self.disable_parameter_updates:
                self.galaxy.get_disk_component(self.disk_components_combobox.currentText()).set_active(state)
            self.auto_generate_galaxy_image()

        self.disk_component_active_checkbox.stateChanged.connect(disk_component_active_checkbox_action)

    def setup_disk_component_emissive_state_adjustment(self):
        self.disk_component_emissive_checkbox = self.window.findChild(QCheckBox, 'diskComponentEmissiveCheckBox')

        quantity_name = 'emissive'
        self.disk_component_parameter_control_tabs[quantity_name] = 'structureTab'
        self.disk_component_parameter_control_widgets[quantity_name] = [self.disk_component_emissive_checkbox]
        self.disk_component_widget_value_setters[quantity_name] = lambda emissive: self.disk_component_emissive_checkbox.setChecked(emissive)

        @Slot(int)
        def disk_component_emissive_checkbox_action(state):
            if not self.disable_parameter_updates:
                self.galaxy.get_disk_component(self.disk_components_combobox.currentText()).set_emission(state)
            self.auto_generate_galaxy_image()

        self.disk_component_emissive_checkbox.stateChanged.connect(disk_component_emissive_checkbox_action)

    def setup_disk_component_slider_spinbox_adjustment(self, quantity_name, widget_name_base, tab, logarithmic=False):
        label = self.window.findChild(QLabel, '{}Label'.format(widget_name_base))
        slider = self.window.findChild(QSlider, '{}Slider'.format(widget_name_base))
        spinbox = self.window.findChild(QDoubleSpinBox, '{}SpinBox'.format(widget_name_base))

        self.disk_component_parameter_control_tabs[quantity_name] = tab
        self.disk_component_parameter_control_widgets[quantity_name] = [label, slider, spinbox]
        self.disk_component_widget_value_setters[quantity_name] = lambda value: spinbox.setValue(value)

        setup_slider_and_spinbox(slider, spinbox,
                                 *galaxy_generation.GalaxyDiskComponent.GUI_param_ranges[quantity_name],
                                 lambda value: self.disk_component_parameter_setter_template(quantity_name, value),
                                 logarithmic=logarithmic)

    def disk_component_parameter_setter_template(self, parameter_name, value):
        if not self.disable_parameter_updates:
            self.galaxy.get_disk_component(self.disk_components_combobox.currentText()).set(parameter_name, value)
        self.auto_generate_galaxy_image()

    def setup_disk_component_strength_scale_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('strength_scale', 'diskComponentStrengthScale', 'structureTab', logarithmic=True)

    def setup_disk_component_disk_extent_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('disk_extent', 'diskComponentDiskExtent', 'structureTab')

    def setup_disk_component_disk_thickness_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('disk_thickness', 'diskComponentDiskThickness', 'structureTab')

    def setup_disk_component_arm_narrowness_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('arm_narrowness', 'diskComponentArmNarrowness', 'structureTab')

    def setup_disk_component_twirl_factor_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('twirl_factor', 'diskComponentTwirlFactor', 'structureTab')

    def setup_disk_component_number_of_octaves_adjustment(self):
        self.disk_component_number_of_octaves_label = self.window.findChild(QLabel, 'diskComponentNumberOfOctavesLabel')
        self.disk_component_number_of_octaves_spinbox = self.window.findChild(QSpinBox, 'diskComponentNumberOfOctavesSpinBox')

        name = 'number_of_octaves'
        self.disk_component_parameter_control_tabs[name] = 'noiseTab'
        self.disk_component_parameter_control_widgets[name] = [self.disk_component_number_of_octaves_label,
                                                               self.disk_component_number_of_octaves_spinbox]

        self.disk_component_widget_value_setters[name] = lambda value: self.disk_component_number_of_octaves_spinbox.setValue(value)

        minimum_number, maximum_number = galaxy_generation.GalaxyDiskComponent.GUI_param_ranges[name]
        self.disk_component_number_of_octaves_spinbox.setMinimum(minimum_number)
        self.disk_component_number_of_octaves_spinbox.setMaximum(maximum_number)

        @Slot(int)
        def disk_component_number_of_octaves_spinbox_action(number):
            if not self.disable_parameter_updates:
                self.galaxy.get_disk_component(self.disk_components_combobox.currentText()).set_number_of_octaves(number)
            self.auto_generate_galaxy_image()

        self.disk_component_number_of_octaves_spinbox.valueChanged.connect(disk_component_number_of_octaves_spinbox_action)

    def setup_disk_component_seed_adjustment(self):
        self.disk_component_seed_label = self.window.findChild(QLabel, 'diskComponentSeedLabel')
        self.disk_component_seed_spinbox = self.window.findChild(QSpinBox, 'diskComponentSeedSpinBox')
        self.disk_component_seed_button = self.window.findChild(QPushButton, 'diskComponentSeedButton')

        name = 'seed'
        self.disk_component_parameter_control_tabs[name] = 'noiseTab'
        self.disk_component_parameter_control_widgets[name] = [self.disk_component_seed_label,
                                                               self.disk_component_seed_spinbox,
                                                               self.disk_component_seed_button]

        self.disk_component_widget_value_setters[name] = lambda value: self.disk_component_seed_spinbox.setValue(value)

        self.disk_component_seed_spinbox.setMaximum(galaxy_generation.GalaxyDiskComponent.get_max_seed())

        @Slot(int)
        def disk_component_seed_spinbox_action(number):
            if not self.disable_parameter_updates:
                self.galaxy.get_disk_component(self.disk_components_combobox.currentText()).set_seed(number)
            self.auto_generate_galaxy_image()

        @Slot()
        def disk_component_seed_button_action():
            disk_component = self.galaxy.get_disk_component(self.disk_components_combobox.currentText())
            disk_component.set_seed(None)
            self.disk_component_seed_spinbox.setValue(disk_component.get_seed())
            self.auto_generate_galaxy_image()

        self.disk_component_seed_spinbox.valueChanged.connect(disk_component_seed_spinbox_action)
        self.disk_component_seed_button.clicked.connect(disk_component_seed_button_action)

    def setup_disk_component_initial_frequency_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('initial_frequency', 'diskComponentInitialFrequency', 'noiseTab')

    def setup_disk_component_lacunarity_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('lacunarity', 'diskComponentLacunarity', 'noiseTab')

    def setup_disk_component_persistence_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('persistence', 'diskComponentPersistence', 'noiseTab')

    def setup_disk_component_noise_threshold_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('noise_threshold', 'diskComponentNoiseThreshold', 'noiseTab')

    def setup_disk_component_noise_cutoff_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('noise_cutoff', 'diskComponentNoiseCutoff', 'noiseTab')

    def setup_disk_component_noise_exponent_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('noise_exponent', 'diskComponentNoiseExponent', 'noiseTab')

    def setup_disk_component_noise_offset_adjustment(self):
        self.setup_disk_component_slider_spinbox_adjustment('noise_offset', 'diskComponentNoiseOffset', 'noiseTab')

    # ****** Bulge component adjustment ******

    def setup_bulge_component_adjustment(self):
        self.setup_bulge_component_parameter_controls()
        self.setup_bulge_component_selection()

    def setup_bulge_component_selection(self):
        self.bulge_component_labels = []
        self.setup_reset_bulge_component_parameters_button()
        self.setup_bulge_components_combobox()
        self.setup_new_bulge_component_button()
        self.setup_remove_bulge_component_button()
        self.setup_bulge_component_name_control()

    def setup_bulge_component_parameter_controls(self):
        self.bulge_component_parameter_control_widgets = {}
        self.bulge_component_widget_value_setters = {}
        self.setup_bulge_component_active_state_adjustment()
        self.setup_bulge_component_emissive_state_adjustment()
        self.setup_bulge_component_strength_scale_adjustment()
        self.setup_bulge_component_bulge_size_adjustment()

    def set_bulge_component_parameter_control_widget_enabled_states(self, enabled):
        for widget_list in self.bulge_component_parameter_control_widgets.values():
            for widget in widget_list:
                widget.setEnabled(enabled)

    def update_bulge_component_parameter_control_widget_values(self, params, excluded_param_names=[], update_bulge_component_parameters=True):
        auto_generate_was_disabled = self.disable_auto_generate
        self.disable_auto_generate = True # Make sure not to recompute the image for each widget

        self.disable_parameter_updates = not update_bulge_component_parameters # We might not want to update the component class attributes automatically

        for name in self.bulge_component_widget_value_setters:
            if name not in excluded_param_names:
                self.bulge_component_widget_value_setters[name](params[name])

        self.disable_parameter_updates = False

        self.disable_auto_generate = auto_generate_was_disabled

    def setup_bulge_components_combobox(self):

        self.bulge_components_combobox = self.window.findChild(QComboBox, 'bulgeComponentsComboBox')

        @Slot(int)
        def bulge_components_combobox_action(index):
            # Update the name text and values of the adjustment controls
            component_label = self.bulge_components_combobox.currentText()
            self.bulge_component_name_line_edit.setText(component_label)
            if len(component_label.strip()) == 0: # Simply disable the relevant widgets if there is no component
                self.set_bulge_component_parameter_control_widget_enabled_states(False)
            else:
                self.set_bulge_component_parameter_control_widget_enabled_states(True) # Make sure control widgets are enabled
                bulge_component = self.galaxy.get_bulge_component(component_label)
                # Update widget values, but make sure the setters don't update the component class attributes, which would be redundant
                self.update_bulge_component_parameter_control_widget_values(bulge_component.get_params(), update_bulge_component_parameters=False)

        self.bulge_components_combobox.currentIndexChanged.connect(bulge_components_combobox_action)
        self.set_bulge_component_parameter_control_widget_enabled_states(False) # Controls are disabled initially

    def setup_new_bulge_component_button(self):

        self.new_bulge_component_button = self.window.findChild(QPushButton, 'newBulgeComponentButton')

        @Slot()
        def new_bulge_component_button_action():
            # Create a new bulge component with a unique label
            component_label = 'Bulge_1'
            while component_label in self.bulge_component_labels: # Keep incrementing the index until the label is unique
                component_label = component_label[:-1] + str(int(component_label[-1]) + 1)

            self.bulge_component_labels.append(component_label)
            self.galaxy.add_bulge_component(galaxy_generation.GalaxyBulgeComponent(component_label))
            self.bulge_components_combobox.addItem(component_label)
            self.bulge_components_combobox.setCurrentIndex(len(self.bulge_component_labels)-1)
            self.auto_generate_galaxy_image()

        self.new_bulge_component_button.clicked.connect(new_bulge_component_button_action)

    def setup_remove_bulge_component_button(self):

        self.remove_bulge_component_button = self.window.findChild(QPushButton, 'removeBulgeComponentButton')

        @Slot()
        def remove_bulge_component_button_action():
            # Remove the currently selected bulge component
            if len(self.bulge_component_labels) == 0: # Nothing to remove if no components are shown
                return
            component_label = self.bulge_components_combobox.currentText()
            self.bulge_component_labels.remove(component_label)
            self.galaxy.remove_bulge_component(component_label)
            self.bulge_components_combobox.removeItem(self.bulge_components_combobox.currentIndex())
            self.auto_generate_galaxy_image()

        self.remove_bulge_component_button.clicked.connect(remove_bulge_component_button_action)

    def setup_bulge_component_name_control(self):

        self.bulge_component_name_line_edit = self.window.findChild(QLineEdit, 'bulgeComponentNameLineEdit')
        self.set_bulge_component_name_button = self.window.findChild(QPushButton, 'setBulgeComponentNameButton')

        @Slot()
        def set_bulge_component_name_button_action():
            # Update the current component label to the content of the line edit
            current_component_label = self.bulge_components_combobox.currentText()
            new_component_label = self.bulge_component_name_line_edit.text()
            if len(new_component_label.strip()) == 0: # Skip updating if the line edit is empty
                return
            if new_component_label in self.bulge_component_labels: # Skip updating if the new label already exists
                return
            component_index = self.bulge_component_labels.index(current_component_label)
            self.bulge_component_labels[component_index] = new_component_label
            self.galaxy.set_bulge_component_label(current_component_label, new_component_label)
            self.bulge_components_combobox.setItemText(self.bulge_components_combobox.currentIndex(), new_component_label)

        self.set_bulge_component_name_button.clicked.connect(set_bulge_component_name_button_action)

    def setup_reset_bulge_component_parameters_button(self):

        self.reset_bulge_component_parameters_button = self.window.findChild(QPushButton, 'resetBulgeComponentParametersButton')

        self.bulge_component_parameter_control_widgets['reset'] = [self.reset_bulge_component_parameters_button]

        @Slot()
        def reset_bulge_component_parameters_button_action():
            # Set the parameters of the currently selected bulge component to the type's default parameters
            component_label = self.bulge_components_combobox.currentText()
            if len(component_label.strip()) == 0: # Skip updating if the line edit is empty
                return
            bulge_component = self.galaxy.get_bulge_component(component_label)
            default_params = bulge_component.__class__.get_default_params()
            # Set widget values to default, which will also update the component class attributes
            self.update_bulge_component_parameter_control_widget_values(default_params, excluded_param_names=['active'])
            self.auto_generate_galaxy_image()

        self.reset_bulge_component_parameters_button.clicked.connect(reset_bulge_component_parameters_button_action)

    def setup_bulge_component_active_state_adjustment(self):
        self.bulge_component_active_checkbox = self.window.findChild(QCheckBox, 'bulgeComponentActiveCheckBox')

        quantity_name = 'active'
        self.bulge_component_parameter_control_widgets[quantity_name] = [self.bulge_component_active_checkbox]
        self.bulge_component_widget_value_setters[quantity_name] = lambda active: self.bulge_component_active_checkbox.setChecked(active)

        @Slot(int)
        def bulge_component_active_checkbox_action(state):
            if not self.disable_parameter_updates:
                self.galaxy.get_bulge_component(self.bulge_components_combobox.currentText()).set_active(state)
            self.auto_generate_galaxy_image()

        self.bulge_component_active_checkbox.stateChanged.connect(bulge_component_active_checkbox_action)

    def setup_bulge_component_emissive_state_adjustment(self):
        self.bulge_component_emissive_checkbox = self.window.findChild(QCheckBox, 'bulgeComponentEmissiveCheckBox')

        quantity_name = 'emissive'
        self.bulge_component_parameter_control_widgets[quantity_name] = [self.bulge_component_emissive_checkbox]
        self.bulge_component_widget_value_setters[quantity_name] = lambda emissive: self.bulge_component_emissive_checkbox.setChecked(emissive)

        @Slot(int)
        def bulge_component_emissive_checkbox_action(state):
            if not self.disable_parameter_updates:
                self.galaxy.get_bulge_component(self.bulge_components_combobox.currentText()).set_emission(state)
            self.auto_generate_galaxy_image()

        self.bulge_component_emissive_checkbox.stateChanged.connect(bulge_component_emissive_checkbox_action)

    def setup_bulge_component_slider_spinbox_adjustment(self, quantity_name, widget_name_base, logarithmic=False):
        label = self.window.findChild(QLabel, '{}Label'.format(widget_name_base))
        slider = self.window.findChild(QSlider, '{}Slider'.format(widget_name_base))
        spinbox = self.window.findChild(QDoubleSpinBox, '{}SpinBox'.format(widget_name_base))

        self.bulge_component_parameter_control_widgets[quantity_name] = [label, slider, spinbox]
        self.bulge_component_widget_value_setters[quantity_name] = lambda value: spinbox.setValue(value)

        setup_slider_and_spinbox(slider, spinbox,
                                 *galaxy_generation.GalaxyBulgeComponent.GUI_param_ranges[quantity_name],
                                 lambda value: self.bulge_component_parameter_setter_template(quantity_name, value),
                                 logarithmic=logarithmic)

    def bulge_component_parameter_setter_template(self, parameter_name, value):
        if not self.disable_parameter_updates:
            self.galaxy.get_bulge_component(self.bulge_components_combobox.currentText()).set(parameter_name, value)
        self.auto_generate_galaxy_image()

    def setup_bulge_component_strength_scale_adjustment(self):
        self.setup_bulge_component_slider_spinbox_adjustment('strength_scale', 'bulgeComponentStrengthScale', logarithmic=True)

    def setup_bulge_component_bulge_size_adjustment(self):
        self.setup_bulge_component_slider_spinbox_adjustment('bulge_size', 'bulgeComponentBulgeSize')

    # ****** Visualization control ******

    def setup_visualization_control(self):
        self.setup_visualization_intensity_control()

    def setup_visualization_intensity_control(self):
        self.gamma = 1/2
        self.brightness = 1
        self.max_intensity = None
        self.setup_visualization_brightness_adjustment()
        self.setup_visualization_gamma_adjustment()

    def setup_visualization_brightness_adjustment(self):

        self.brightness_slider = self.window.findChild(QSlider, 'brightnessSlider')
        self.brightness_spinbox = self.window.findChild(QDoubleSpinBox, 'brightnessSpinBox')

        def brightness_setter(value):
            self.brightness = value
            self.auto_generate_galaxy_image()

        setup_slider_and_spinbox(self.brightness_slider, self.brightness_spinbox, 0.05, 5, 0.05, brightness_setter)

        self.brightness_spinbox.setValue(self.brightness)

        self.setup_fix_brightness_reference_checkbox()

    def setup_fix_brightness_reference_checkbox(self):
        self.fix_brightness_reference_checkbox = self.window.findChild(QCheckBox, 'fixBrightnessReferenceCheckBox')
        self.fix_brightness_reference_checkbox.setChecked(False)

    def setup_visualization_gamma_adjustment(self):

        self.gamma_slider = self.window.findChild(QSlider, 'gammaSlider')
        self.gamma_spinbox = self.window.findChild(QDoubleSpinBox, 'gammaSpinBox')

        def gamma_setter(value):
            self.gamma = value
            self.auto_generate_galaxy_image()

        setup_slider_and_spinbox(self.gamma_slider, self.gamma_spinbox, 0.1, 2, 0.02, gamma_setter)

        self.gamma_spinbox.setValue(self.gamma)

    def convert_intensity_to_image_values(self, intensity):
        if self.max_intensity is None or not self.fix_brightness_reference_checkbox.isChecked():
            self.max_intensity = np.max(intensity)
        image_values = image_utils.perform_liear_stretch(intensity, 0, self.max_intensity/self.brightness)
        return image_utils.perform_gamma_correction(image_values, self.gamma)


def setup_slider_and_spinbox(slider, spinbox, minimum_value, maximum_value, step, value_setter, logarithmic=False):

    minimum_slider_value = 1
    maximum_slider_value = 1000
    slider.setMinimum(minimum_slider_value)
    slider.setMaximum(maximum_slider_value)
    spinbox.setMinimum(minimum_value)
    spinbox.setMaximum(maximum_value)
    spinbox.setSingleStep(step)

    if logarithmic:
        assert minimum_value > 0 and maximum_value > 0

    def to_fraction(value, minimum, maximum):
        return (value - minimum)/(maximum - minimum)

    def from_fraction(value, minimum, maximum):
        return minimum + (maximum - minimum)*value

    def slider_value_from_slider_fraction(slider_fraction):
        return int(round(from_fraction(slider_fraction, minimum_slider_value, maximum_slider_value)))

    def slider_fraction_from_slider_value(slider_value):
        return to_fraction(slider_value, minimum_slider_value, maximum_slider_value)

    def linear_spinbox_value_from_slider_fraction(slider_fraction):
        return from_fraction(slider_fraction, minimum_value, maximum_value)

    def logarithmic_spinbox_value_from_slider_fraction(slider_fraction):
        return 10**from_fraction(slider_fraction, np.log10(minimum_value), np.log10(maximum_value))

    def linear_slider_fraction_from_spinbox_value(spinbox_value):
        return to_fraction(spinbox_value, minimum_value, maximum_value)

    def logarithmic_slider_fraction_from_spinbox_value(spinbox_value):
        return to_fraction(np.log10(spinbox_value), np.log10(minimum_value), np.log10(maximum_value))

    @Slot(int)
    def set_spinbox_value(slider_value):
        slider_fraction = slider_fraction_from_slider_value(slider_value)
        spinbox_value = logarithmic_spinbox_value_from_slider_fraction(slider_fraction) if logarithmic else linear_spinbox_value_from_slider_fraction(slider_fraction)
        spinbox.setValue(spinbox_value)
        value_setter(spinbox_value)

    @Slot(float)
    def set_slider_value(spinbox_value):
        slider_fraction = logarithmic_slider_fraction_from_spinbox_value(spinbox_value) if logarithmic else linear_slider_fraction_from_spinbox_value(spinbox_value)
        slider_value = slider_value_from_slider_fraction(slider_fraction)
        if slider_value != slider.value():
            slider.setValue(slider_value)

    slider.valueChanged.connect(set_spinbox_value)
    spinbox.valueChanged.connect(set_slider_value)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    sys.exit(app.exec_())
