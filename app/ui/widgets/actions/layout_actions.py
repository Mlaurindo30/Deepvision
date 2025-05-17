from typing import TYPE_CHECKING
from functools import partial

from PySide6 import QtWidgets, QtCore
if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets import widget_components
from app.helpers.typing_helper import LayoutDictTypes

def safe_int(val):
    try:
        return int(float(val))
    except Exception:
        return 0

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return 0.0

def add_widgets_to_tab_layout(main_window: 'MainWindow', LAYOUT_DATA: LayoutDictTypes, layoutWidget: QtWidgets.QVBoxLayout, data_type='parameter'):
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(0, 0, 10, 0)
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll_content = QtWidgets.QWidget()
    scroll_content.setLayout(layout)
    scroll_area.setWidget(scroll_content)
    scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

    def add_horizontal_layout_to_category(category_layout: QtWidgets.QFormLayout, *widgets):
        horizontal_layout = QtWidgets.QHBoxLayout()
        for widget in widgets:
            horizontal_layout.addWidget(widget)
        category_layout.addRow(horizontal_layout)
        return horizontal_layout

    for category, widgets in LAYOUT_DATA.items():
        group_box = widget_components.FormGroupBox(main_window, title=main_window.tr(category))
        main_window.register_dynamic_widget(group_box, category)
        category_layout = QtWidgets.QFormLayout()
        category_layout.setVerticalSpacing(6)
        category_layout.setHorizontalSpacing(8)
        group_box.setLayout(category_layout)

        for widget_name, widget_data in widgets.items():
            spacing_level = widget_data.get('level', 0)
            label_key = widget_data['label']
            help_key = widget_data['help']
            label = QtWidgets.QLabel(main_window.tr(label_key))
            label.setToolTip(main_window.tr(help_key))
            label.setWordWrap(True)
            label.setMinimumWidth(130)
            label.setMaximumWidth(200)
            label.translation_key = label_key
            main_window.register_dynamic_widget(label, label_key)
            horizontal_layout = None

            if 'Toggle' in widget_name:
                widget = widget_components.ToggleButton(
                    label=main_window.tr(label_key),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    main_window=main_window)
                widget.setChecked(widget_data['default'])
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)
                widget.reset_default_button.setToolTip(main_window.tr('Resetar'))
                main_window.register_dynamic_widget(widget.reset_default_button, 'Resetar')
                horizontal_layout = add_horizontal_layout_to_category(category_layout, widget, label, widget.reset_default_button)
                if data_type=='parameter':
                    common_widget_actions.create_default_parameter(main_window, widget_name, widget_data['default'])
                else:
                    common_widget_actions.create_control(main_window, widget_name, widget_data['default'])
                def onchange(toggle_widget, toggle_widget_name, widget_data, *args):
                    toggle_state = toggle_widget.isChecked()
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, toggle_widget_name, toggle_state, enable_refresh_frame=getattr(toggle_widget, 'enable_refresh_frame', True), exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, toggle_widget_name, toggle_state, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.toggled.connect(partial(onchange, widget, widget_name, widget_data))

            elif 'Selection' in widget_name:
                widget = widget_components.SelectionBox(
                    label=main_window.tr(label_key),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    main_window=main_window,
                    default_value=widget_data['default'],
                    selection_values=widget_data['options']
                )
                main_window.register_dynamic_widget(widget, label_key)
                if callable(widget_data['options']):
                    widget.addItems(widget_data['options']())
                    widget.setCurrentText(widget_data['default']())
                else:
                    widget.addItems(widget_data['options'])
                    widget.setCurrentText(widget_data['default'])
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)
                widget.reset_default_button.setToolTip(main_window.tr('Resetar'))
                main_window.register_dynamic_widget(widget.reset_default_button, 'Resetar')
                horizontal_layout = add_horizontal_layout_to_category(category_layout, label, widget, widget.reset_default_button)
                if data_type=='parameter':
                    common_widget_actions.create_default_parameter(main_window, widget_name, widget_data['default'] if not callable(widget_data['default']) else widget_data['default']())
                else:
                    common_widget_actions.create_control(main_window, widget_name, widget_data['default'] if not callable(widget_data['default']) else widget_data['default']())
                def onchange(selection_widget, selection_widget_name, widget_data, selected_value=False):
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, selection_widget_name, selected_value, enable_refresh_frame=getattr(selection_widget, 'enable_refresh_frame', True))
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, selection_widget_name, selected_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.currentTextChanged.connect(partial(onchange, widget, widget_name, widget_data))

            elif 'DecimalSlider' in widget_name:
                widget = widget_components.ParameterDecimalSlider(
                    label=main_window.tr(label_key),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    min_value=safe_float(widget_data['min_value']),
                    max_value=safe_float(widget_data['max_value']),
                    default_value=safe_float(widget_data['default']),
                    decimals=int(widget_data['decimals']),
                    step_size=safe_float(widget_data['step']),
                    main_window=main_window
                )
                main_window.register_dynamic_widget(widget, label_key)
                widget.line_edit = widget_components.ParameterLineDecimalEdit(
                    min_value=safe_float(widget_data['min_value']),
                    max_value=safe_float(widget_data['max_value']),
                    default_value=str(widget_data['default']),
                    decimals=int(widget_data['decimals']),
                    step_size=safe_float(widget_data['step']),
                    fixed_width=48,
                    max_length=7 if int(widget_data['decimals']) > 1 else 5
                )
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)
                widget.reset_default_button.setToolTip(main_window.tr('Resetar'))
                main_window.register_dynamic_widget(widget.reset_default_button, 'Resetar')
                horizontal_layout = add_horizontal_layout_to_category(category_layout, label, widget, widget.line_edit, widget.reset_default_button)
                if data_type=='parameter':
                    common_widget_actions.create_default_parameter(main_window, widget_name, safe_float(widget_data['default']))
                else:
                    common_widget_actions.create_control(main_window, widget_name, safe_float(widget_data['default']))
                def onchange_slider(slider_widget, slider_widget_name, widget_data, new_value=False):
                    actual_value = slider_widget.value()
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, slider_widget_name, actual_value, enable_refresh_frame=getattr(slider_widget, 'enable_refresh_frame', True))
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, slider_widget_name, actual_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                    slider_widget.line_edit.setText(str(actual_value))
                widget.debounce_timer.timeout.connect(partial(onchange_slider, widget, widget_name, widget_data))
                def onchange_line_edit(slider_widget, slider_widget_name, widget_data, new_value=False):
                    if not new_value:
                        new_value = 0.0
                    try:
                        new_value = float(new_value)
                    except Exception:
                        new_value = slider_widget.value()
                    if new_value > (slider_widget.max_value / slider_widget.scale_factor):
                        new_value = slider_widget.max_value / slider_widget.scale_factor
                    elif new_value < (slider_widget.min_value / slider_widget.scale_factor):
                        new_value = slider_widget.min_value / slider_widget.scale_factor
                    slider_widget.setValue(new_value)
                    slider_widget.line_edit.setText(str(new_value))
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, slider_widget_name, new_value, enable_refresh_frame=getattr(slider_widget, 'enable_refresh_frame', True))
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, slider_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.line_edit.textChanged.connect(partial(onchange_line_edit, widget, widget_name, widget_data))

            elif 'Slider' in widget_name:
                if widget_name == 'BenchmarkThreadsButton':
                    continue
                minv = safe_int(widget_data['min_value'])
                maxv = safe_int(widget_data['max_value'])
                defv = safe_int(widget_data['default'])
                widget = widget_components.ParameterSlider(
                    label=main_window.tr(label_key),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    min_value=minv,
                    max_value=maxv,
                    default_value=defv,
                    step_size=safe_int(widget_data['step']),
                    main_window=main_window
                )
                main_window.register_dynamic_widget(widget, label_key)
                widget.line_edit = widget_components.ParameterLineEdit(
                    min_value=minv, max_value=maxv, default_value=str(defv)
                )
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)
                widget.reset_default_button.setToolTip(main_window.tr('Resetar'))
                main_window.register_dynamic_widget(widget.reset_default_button, 'Resetar')
                horizontal_layout = add_horizontal_layout_to_category(category_layout, label, widget, widget.line_edit, widget.reset_default_button)
                if data_type=='parameter':
                    common_widget_actions.create_default_parameter(main_window, widget_name, defv)
                else:
                    common_widget_actions.create_control(main_window, widget_name, defv)
                def onchange_slider(slider_widget, slider_widget_name, widget_data, new_value=False):
                    value = slider_widget.value()
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, slider_widget_name, value, enable_refresh_frame=getattr(slider_widget, 'enable_refresh_frame', True))
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, slider_widget_name, value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                    slider_widget.line_edit.setText(str(value))
                widget.debounce_timer.timeout.connect(partial(onchange_slider, widget, widget_name, widget_data))
                def onchange_line_edit(slider_widget, slider_widget_name, widget_data, new_value=False):
                    if not new_value:
                        new_value = 0
                    try:
                        new_value = int(float(new_value))
                    except Exception:
                        new_value = slider_widget.value()
                    if new_value > slider_widget.max_value:
                        new_value = slider_widget.max_value
                    elif new_value < slider_widget.min_value:
                        new_value = slider_widget.min_value
                    slider_widget.setValue(int(new_value))
                    slider_widget.line_edit.setText(str(new_value))
                    if data_type=='parameter':
                        common_widget_actions.update_parameter(main_window, slider_widget_name, new_value, enable_refresh_frame=getattr(slider_widget, 'enable_refresh_frame', True))
                    elif data_type=='control':
                        common_widget_actions.update_control(main_window, slider_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.line_edit.textChanged.connect(partial(onchange_line_edit, widget, widget_name, widget_data))

            elif 'Text' in widget_name:
                widget = widget_components.ParameterText(
                    label=main_window.tr(label_key),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    default_value=str(widget_data['default']),
                    fixed_width=widget_data.get('width', 120),
                    main_window=main_window,
                    data_type=data_type,
                    exec_function=widget_data.get('exec_function'),
                    exec_function_args=widget_data.get('exec_function_args', [])
                )
                main_window.register_dynamic_widget(widget, label_key)
                widget.reset_default_button = widget_components.ParameterResetDefaultButton(related_widget=widget)
                widget.reset_default_button.setToolTip(main_window.tr('Resetar'))
                main_window.register_dynamic_widget(widget.reset_default_button, 'Resetar')
                horizontal_layout = add_horizontal_layout_to_category(category_layout, label, widget, widget.reset_default_button)
                if data_type=='parameter':
                    common_widget_actions.create_default_parameter(main_window, widget_name, widget_data['default'])
                else:
                    common_widget_actions.create_control(main_window, widget_name, widget_data['default'])
                def on_enter_pressed(text_widget, text_widget_name):
                    new_value = text_widget.text()
                    if data_type == 'parameter':
                        common_widget_actions.update_parameter(main_window, text_widget_name, new_value, enable_refresh_frame=getattr(text_widget, 'enable_refresh_frame', True))
                    else:
                        common_widget_actions.update_control(main_window, text_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.returnPressed.connect(partial(on_enter_pressed, widget, widget_name))

            elif 'Button' in widget_name:
                widget = QtWidgets.QPushButton(main_window.tr(label_key))
                main_window.register_dynamic_widget(widget, label_key)
                widget.setToolTip(main_window.tr(help_key))
                widget.group_layout_data = widgets
                widget.setFixedWidth(180)
                horizontal_layout = QtWidgets.QHBoxLayout()
                horizontal_layout.addStretch()
                horizontal_layout.addWidget(widget)
                category_layout.addRow(horizontal_layout)
                def on_button_clicked(btn_widget, btn_widget_name, btn_widget_data):
                    if btn_widget_data.get('exec_function'):
                        btn_widget_data['exec_function'](main_window)
                widget.clicked.connect(partial(on_button_clicked, widget, widget_name, widget_data))
                main_window.parameter_widgets[widget_name] = widget

            if horizontal_layout:
                horizontal_layout.setContentsMargins(spacing_level * 10, 2, 2, 2)
            main_window.parameter_widgets[widget_name] = widget

        layout.addWidget(group_box)
    layoutWidget.addWidget(scroll_area)
    for category, widgets in LAYOUT_DATA.items():
        for widget_name, widget_data in widgets.items():
            widget = main_window.parameter_widgets[widget_name]
            common_widget_actions.show_hide_related_widgets(main_window, widget, widget_name)

def show_hide_faces_panel(main_window: 'MainWindow', checked):
    if checked:
        main_window.facesPanelGroupBox.show()
    else:
        main_window.facesPanelGroupBox.hide()
    fit_image_to_view_onchange(main_window)

def show_hide_input_target_media_panel(main_window: 'MainWindow', checked):
    if checked:
        main_window.input_Target_DockWidget.show()
    else:
        main_window.input_Target_DockWidget.hide()
    fit_image_to_view_onchange(main_window)

def show_hide_parameters_panel(main_window: 'MainWindow', checked):
    if checked:
        main_window.controlOptionsDockWidget.show()
    else:
        main_window.controlOptionsDockWidget.hide()
    fit_image_to_view_onchange(main_window)

def fit_image_to_view_onchange(main_window: 'MainWindow', *args):
    pixmap_items = main_window.scene.items()
    if pixmap_items:
        pixmap_item = pixmap_items[0]
        scene_rect = pixmap_item.boundingRect()
        QtCore.QTimer.singleShot(0, partial(graphics_view_actions.fit_image_to_view, main_window, pixmap_item, scene_rect))

def set_up_menu_actions(main_window: 'MainWindow'):
    main_window.actionLoad_SavedWorkspace.triggered.connect(partial(save_load_actions.load_saved_workspace, main_window,))
    main_window.actionSave_CurrentWorkspace.triggered.connect(partial(save_load_actions.save_current_workspace, main_window,))

    main_window.actionOpen_Videos_Folder.triggered.connect(partial(list_view_actions.select_target_medias, main_window, 'folder'))
    main_window.actionOpen_Video_Files.triggered.connect(partial(list_view_actions.select_target_medias, main_window, 'files'))
    main_window.actionLoad_Source_Image_Files.triggered.connect(partial(list_view_actions.select_input_face_images, main_window, 'files'))
    main_window.actionLoad_Source_Images_Folder.triggered.connect(partial(list_view_actions.select_input_face_images, main_window, 'folder'))
    main_window.actionLoad_Embeddings.triggered.connect(partial(save_load_actions.open_embeddings_from_file, main_window))
    main_window.actionSave_Embeddings.triggered.connect(partial(save_load_actions.save_embeddings_to_file, main_window))
    main_window.actionSave_Embeddings_As.triggered.connect(partial(save_load_actions.save_embeddings_to_file, main_window))
    main_window.actionView_Fullscreen_F11.triggered.connect(partial(video_control_actions.view_fullscreen, main_window))

def disable_all_parameters_and_control_widget(main_window: 'MainWindow'):
    # Disable all bottom buttons
    main_window.saveImageButton.setDisabled(True)
    main_window.findTargetFacesButton.setDisabled(True)
    main_window.clearTargetFacesButton.setDisabled(True)
    main_window.swapfacesButton.setDisabled(True)
    main_window.editFacesButton.setDisabled(True)
    main_window.openEmbeddingButton.setDisabled(True)
    main_window.saveEmbeddingButton.setDisabled(True)
    main_window.saveEmbeddingAsButton.setDisabled(True)

    # Disable all video control buttons
    main_window.videoSeekSlider.setDisabled(True)
    main_window.addMarkerButton.setDisabled(True)
    main_window.removeMarkerButton.setDisabled(True)
    main_window.nextMarkerButton.setDisabled(True)
    main_window.previousMarkerButton.setDisabled(True)
    main_window.frameAdvanceButton.setDisabled(True)
    main_window.frameRewindButton.setDisabled(True)

    # Disable compare checkboxes
    main_window.faceCompareCheckBox.setDisabled(True)
    main_window.faceMaskCheckBox.setDisabled(True)

    # Disable list items
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.setDisabled(True)
    for _, target_media_button in main_window.target_videos.items():
        target_media_button.setDisabled(True)
    for _, input_face_button in main_window.input_faces.items():
        input_face_button.setDisabled(True)
    for _, target_face_button in main_window.target_faces.items():
        target_face_button.setDisabled(True)

    # Disable parameters and controls dict widgets
    for _, widget in main_window.parameter_widgets.items():
        widget.setDisabled(True)
        if hasattr(widget, 'reset_default_button'):
            widget.reset_default_button.setDisabled(True)
        if hasattr(widget, 'label_widget'):
            widget.label_widget.setDisabled(True)
        if hasattr(widget, 'line_edit') and widget.line_edit:
            widget.line_edit.setDisabled(True)
