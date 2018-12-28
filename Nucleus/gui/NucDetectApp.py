'''
Created on 11.11.2018

@author: Romano Weiss
'''
import os
import matplotlib
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.settings import Settings
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, ListProperty, BooleanProperty
from kivy.uix.popup import Popup
from kivy.factory import Factory
from NucDetect.core.Detector import Detector
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.behaviors.focus import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.button import Label
from threading import Thread
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')


class NucDetect(App):

    def build(self):
        self.title = "NucDetect -- Alpha"
        self.icon = "logo.png"
        self.settings_cls = Settings
        self.use_kivy_settings = False
        self.controller = Controller()
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"images")
        self.controller.add_images_from_folder(pathpardir)
        return self.controller

    def build_config(self, config):
        config.read("settings.ini")
        App.build_config(self, config)

    def on_config_change(self, config, section, key, value):
        if config is self.config:
            print("Config changed!")
            pass

    def build_settings(self, settings):
        settings.add_json_panel("General Settings", self.config,
                                filename="settings_general.json")
        settings.add_json_panel("Analysis", self.config,
                                filename="settings_analysis.json")
        settings.pos_hint = ({"center_x": .5, "center_y": .5})
        settings.size_hint = (0.9, 0.9)
        App.build_settings(self, settings)

    def open_file(self, path, filename):
        self.images.append(filename)
        self.dismiss_popup()
        self.controller.add_image_to_list(filename[0], path)

    def show_load_dialog(self):
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"images")
        os.makedirs(pathpardir, exist_ok=True)
        content = OpenFileDialog(load=self.open_file,
                                 cancel=self.dismiss_popup)
        content.path = pathpardir
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def save_results(self):
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results")
        os.makedirs(pathpardir, exist_ok=True)
        pathresult = os.path.join(pathpardir,
                                  "result - " + self.id + ".csv")
        pass

    def dismiss_popup(self):
        self._popup.dismiss()


class SaveFileDialog(FloatLayout):
    save = ObjectProperty()
    text_input = ObjectProperty()
    cancel = ObjectProperty()


class OpenFileDialog(FloatLayout):
    load = ObjectProperty()
    cancel = ObjectProperty()
    path = StringProperty()


class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):

        def __init__(self, **kwargs):
            super(SelectableRecycleBoxLayout, self).__init__(**kwargs)


class TableView(BoxLayout):
    has_header = BooleanProperty()
    rows = NumericProperty()
    cols = NumericProperty()
    selectable_cells = BooleanProperty(True)
    callback = None

    def __init__(self, **kwargs):
        super(TableView, self).__init__(**kwargs)

    def prepare_table(self, cols=1, rows=1):
        self.cols = cols
        self.rows = rows
        for x in range(cols):
            t = TableColumn()
            t.is_selectable = self.selectable_cells
            t.has_header = self.has_header
            t.set_row_count(rows)
            self.add_widget(t)
        self._prepare_columns()

    def _prepare_columns(self):
        for x in reversed(range(len(self.children)-1)):
            self.children[x].bar_width = 0
        self.children.reverse()

    def set_data(self, col, row, data, add_data={}):
        data.reverse()
        if col > self.cols:
            for x in range(len(self.children), col):
                t = TableColumn()
                t.set_row_count(self.rows)
                t.is_selectable = self.selectable_cells
                t.has_header = self.has_header
                self.add_widget(t)
                self.children[x].set_row_count(row)
            self._prepare_columns()
        if row > self.rows:
            for column in self.children:
                column.set_row_count(row)
        self.children[col].add_data(row, data, add_data)

    def add_data(self, data, add_data={}, row=None):
        data.reverse()
        if len(data) > self.cols:
            self.cols = len(data)
            ind = 0
            for col in self.children:
                col.add_data(data[ind], add_data)
                ind += 1
            for x in range(len(data)-ind):
                t = TableColumn()
                t.set_row_count(self.rows)
                t.is_selectable = self.selectable_cells
                t.has_header = self.has_header
                t.add_data(data[x+ind], add_data)
                self.add_widget(t)
            self._prepare_columns()
        elif row is None:
            for num in range(len(data)):
                temp = {}
                if len(add_data) is not 0:
                    temp = add_data[num]
                self.children[num].add_data(data=data[num], add_data=temp)
        elif row > self.rows:
            self.rows = row
            for col in range(self.cols):
                col.set_row_count(row)
            for col in range(len(data)):
                temp = {}
                if len(add_data) is not 0:
                    temp = add_data[col]
                self.add_data(col, row, data=data[col], add_data=temp)
        else:
            for num in range(len(data)):
                temp = {}
                if len(add_data) is not 0:
                    temp = add_data[num]
                col.set_data(row, data=data[num], add_data=temp)

    def add_header(self, header):
        self.add_data_line(0, header)

    def set_header_color(self, color):
        for col in self.children:
            col.header_col = color

    def get_data(self, row, col):
        if col > self.cols or row > self.rows:
            raise IndexError()
        else:
            return self.children[col].get_data(row)

    def select_column(self, col):
        self.children[col].select_all()

    def register_selection(self, data, add_data):
        if self.callback is not None:
            self.callback.register_selection(data, add_data)

    def register_scrolling(self, instance, value):
        for col in self.children:
            if col is not instance:
                col.scroll_y = value

    def clear_data(self):
        for col in self.children:
            col.clear_data()


class TableColumn(RecycleView):
    has_header = BooleanProperty()
    background = ObjectProperty((0, 0, 0, 0))
    is_selectable = BooleanProperty(True)
    callback = None
    last_index = 0

    def __init__(self, **kwargs):
        super(TableColumn, self).__init__(**kwargs)

    def add_data(self, data, add_data={}):
        ind = min(self.last_index, len(self.data)-1)
        if self.data[ind]["text"] is "":
            self.data[ind]["text"] = str(data)
            self.data[ind]["selectable"] = self.is_selectable
            self.data[ind]["selected"] = False
            self.data[ind]["add_data"] = add_data
            if self.has_header:
                self.data[ind]["is_header"] = self.has_header
            self.last_index += 1
        else:
            self.data.append({"text": str(data),
                              "add_data": add_data,
                              "selectable": self.is_selectable,
                              "selected": False,
                              "is_header": self.has_header})
            self.last_index += 1

    def set_data(self, row, data, add_data={}):
        self.data[row]["text"] = data
        if len(add_data) is not 0:
            self.data[row]["add_data"] = add_data
        else:
            self.data[row]["add_data"] = {}

    def set_row_count(self, rows):
        for x in range(rows):
            self.data.append({"text": "", "add_data": {}})

    def get_data(self, row):
        if row > len(self.data):
            raise IndexError("Index out of range!")
        else:
            return self.data[row]["text"], self.data[row]["add_data"]

    def register_selection(self, data, add_data):
        self.parent.register_selection(data, add_data)

    def on_scroll_y(self, instance, value):
        self.parent.register_scrolling(instance, value)

    def clear_data(self):
        self.last_index = 0
        for cell in self.data:
            cell["text"] = ""
            cell["add_data"] = {}


class TableCell(RecycleDataViewBehavior, Label):
    background = ObjectProperty((.30, .30, .30, .50))
    select_col = ObjectProperty((.25, .25, .40, .65))
    header_col = ObjectProperty((.25, .25, .40, .50))
    index = None
    is_header = BooleanProperty(False)
    add_data = ObjectProperty()
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(TableCell, self).__init__(**kwargs)

    def refresh_view_attrs(self, rv, index, data):
        ''' Catch and handle the view changes '''
        self.index = index
        return super(TableCell, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        ''' Add selection on touch down '''
        if super(TableCell, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        ''' Respond to the selection of items in the view. '''
        if self.parent is not None:
            if self.selectable:
                self.selected = is_selected
            if is_selected:
                self.parent.parent.register_selection(self.text, self.add_data)


class ResultWindow(BoxLayout):
    figure_window = ObjectProperty(None)
    image_key = StringProperty(None)
    detector = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(ResultWindow, self).__init__(**kwargs)

    def add_figure(self, key):
        self.image_key = key
        figure = self.detector.get_result_image_as_figure(key)
        self.cnvs = FigureCanvasKivyAgg(figure=figure)
        self.figure_window.add_widget(self.cnvs)

    def change_figure(self, id):
        fig = None
        plt.clf()
        snaps = self.detector.get_snapshot(self.image_key)
        if id == "red_channel":
            fig = self._change_figure(snaps["channel"][1], gray=False)
        elif id == "green_channel":
            fig = self._change_figure(snaps["channel"][2], gray=False)
        elif id == "blue_channel":
            fig = self._change_figure(snaps["channel"][0], gray=False)
        elif id == "red_thr":
            fig = self._change_figure(snaps["binarized"][1])
        elif id == "green_thr":
            fig = self._change_figure(snaps["binarized"][2])
        elif id == "blue_thr":
            fig = self._change_figure(snaps["binarized"][0])
        elif id == "red_marker":
            fig = self._change_figure(snaps["watershed"][1])
        elif id == "green_marker":
            fig = self._change_figure(snaps["watershed"][2])
        elif id == "blue_marker":
            fig = self._change_figure(snaps["watershed"][0])
        elif id == "result":
            fig = self.detector.get_result_image_as_figure(
                self.image_key
            )
        elif id == "original":
            fig = self._change_figure(snaps["original"])
        self.cnvs.figure = fig
        self.cnvs.draw()
        self.canvas.ask_update()

    def _change_figure(self, img_array, gray=True):
        result = plt.figure()
        ax = result.add_subplot(111)
        if gray:
            ax.imshow(img_array, cmap="gray")
        else:
            ax.imshow(img_array)
        return result


class AnalyserThread(Thread):
    callback = None
    key = ""

    def __init__(self, target=None, callback=None, key="", *args, **kwargs):
        super(AnalyserThread, self).__init__(target=target, *args, **kwargs)
        self.callback = callback
        self.key = key

    def run(self):
        self._target(self.key)
        if self.callback is not None:
            self.callback()


class Controller(FloatLayout):
    result_table = ObjectProperty(None)
    image_list = ObjectProperty(None)
    prg_bar = ObjectProperty(None)
    prg_bar_label = ObjectProperty(None)
    show_btn = ObjectProperty(None)
    save_btn = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.image_list.prepare_table()
        self.image_list.callback = self
        self.result_table.prepare_table(cols=6, rows=10)
        self.detector = Detector()
        self.reg_image = {}
        self.prg_bar.max = 100

    def add_image_to_list(self, name, path):
        file = path[path.rfind("\\")+1:]
        self.image_list.add_data(data=[file],
                                 add_data=[{"file": file, "path": path}])
        self.detector.load_image(name)

    def add_images_from_folder(self, url):
        files = os.listdir(url)
        for file in files:
            path = os.path.join(url, file)
            if os.path.isfile(path):
                file = path[path.rfind("\\")+1:]
                self.image_list.add_data(data=[file],
                                         add_data=[{"file": file,
                                                    "path": path}]
                                         )
        self.select_all_images()

    def remove_image_from_list(self, widget):
        self.image_list.remove_widget(widget)

    def analyze(self):
        if not self.reg_image:
            print("No image selected!")
        else:
            print("Start analysis")
            self.set_progress("Analysing " + str(self.reg_image["path"]),
                              0)
            key = self.reg_image["key"]
            thread = AnalyserThread(target=self.analyze_image, key=key)
            thread.start()

    def analyze_image(self, key):
        self.detector.analyse_image(key)
        data = self.detector.get_output(key)
        self.result_table.clear_data()
        self.result_table.add_data(data=data["header"])
        for x in range(len(data["data"])):
            dat = data["data"][x]
            self.result_table.add_data(data=dat)
        self.set_progress("Analysis finished -- Program ready", 100)
        self.save_btn.disabled = False
        self.show_btn.disabled = False

    def set_progress(self, text, progress):
        self.prg_bar_label.text = text
        self.prg_bar.value = progress

    def analyze_all(self):
        pass

    def register_selection(self, data, add_data):
        self.reg_image["path"] = add_data["path"]
        self.reg_image["key"] = self.detector.load_image(add_data["path"])

    def select_all_images(self):
        '''
        self.image_list.select_column(0)
        '''

    def show_result_image(self):
        rw = ResultWindow()
        rw.detector = self.detector
        rw.add_figure(self.reg_image["key"])
        pop = Popup(title="Result Image", content=rw,size_hint=(0.9, 0.9))
        pop.open()

    def save_results(self):
        key = self.reg_image["key"]
        save = Thread(target=self._save_results, args=(key,))
        self.set_progress("Saving Results", 0)
        save.start()

    def _save_results(self, key):
        self.detector.create_ouput(key)
        self.set_progress("Saving Results", 50)
        self.detector.save_result_image(key)
        self.set_progress("Results saved -- Program ready", 100)


Factory.register('Root', cls=Controller)
Factory.register('LoadDialog', cls=OpenFileDialog)
Factory.register('SaveDialog', cls=SaveFileDialog)

if __name__ == '__main__':
    NucDetect().run()
