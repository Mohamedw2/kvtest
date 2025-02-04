from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import qrcode
from kivy.uix.textinput import TextInput
from kivymd.uix.textfield import MDTextField
from kivy.uix.scrollview import ScrollView
from pathlib import Path
from datetime import datetime
import logging
import os
from kivy.clock import Clock
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
import pandas as pd
from plyer import camera
from kivymd.uix.selectioncontrol import MDCheckbox
from kivy.graphics import Color, Rectangle, Line
from kivy.uix.widget import Widget
from kivymd.uix.button import MDRectangleFlatButton
from kivy.uix.filechooser import FileChooserListView
from shutil import copyfile
import cv2
from pyzbar.pyzbar import decode
from PIL import Image as PILImage
from datetime import datetime, timedelta
from kivymd.uix.menu import MDDropdownMenu
from kivy.graphics.texture import Texture
import logging
import warnings
import numpy as np

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ignore specific warnings from pyzbar if they are not critical
warnings.filterwarnings("ignore", category=Warning, module="pyzbar")

# Get the directory of the script
SCRIPT_DIR = Path(__file__).parent


# Create application directory in user's home directory
APP_DIR = SCRIPT_DIR
DATA_FILE = APP_DIR / "data" / "data.xlsx"
PHOTOS_DIR = APP_DIR / "photos"
QR_CODES_DIR = APP_DIR / "qr_codes"

# Ensure directories exist
(APP_DIR / "data").mkdir(exist_ok=True)
PHOTOS_DIR.mkdir(exist_ok=True)
QR_CODES_DIR.mkdir(exist_ok=True)

class ImageButton(ButtonBehavior, Image):
    pass

# Initialize Excel file if it doesn't exist
if not DATA_FILE.exists():
    df = pd.DataFrame(columns=["Name", "Phone", "ID Number", "Photo", "Date", "Attendance"])
    df.to_excel(DATA_FILE, index=False)

class MainScreen(Screen):
    def go_to_scan_screen(self):
        self.manager.current = 'scan'

class RegisterScreen(Screen):
    def generate_qr(self):
        try:
            name = self.ids.name.text
            id_number = self.ids.id_number.text
            day = self.ids.day.text
            month = self.ids.month.text
            year = self.ids.year.text
            
            if not all([name, id_number, day, month, year]) or not hasattr(self, 'file_name'):
                self.show_popup("Error", "Please fill all fields and select a photo!")
                return

            date = f"{day}-{month}-{year}"
            input_date = datetime.strptime(date, "%d-%m-%Y")
            current_date = datetime.now()

            if input_date < current_date:
                attendance_status = "Old Date"
                status_color = (1, 0, 0)  # Red
            elif input_date > current_date:
                attendance_status = "Invited"
                status_color = (0.5, 0.5, 0.5)  # Light Gray
            else:
                attendance_status = "Present"
                status_color = (0, 1, 0)  # Green

            # Generate QR Code with only name and ID number
            data = f"{name},{id_number}"
            print(f"QR Code Data: {data}")  # Debug: Print QR code data
            qr = qrcode.make(data)
            qr_path = QR_CODES_DIR / f"{name}_{id_number}.png"
            qr.save(str(qr_path))

            # Read existing data or create a new DataFrame if it doesn't exist
            df = pd.read_excel(DATA_FILE) if DATA_FILE.exists() else pd.DataFrame(
                columns=["Name", "Phone", "ID Number", "Photo", "Date", "Attendance"])
            
            logger.debug(f"Current DataFrame before adding new row: {df.to_dict()}")
            
            # Add new row
            new_row = {
                "Name": name,
                "Phone": self.ids.phone.text,
                "ID Number": id_number,
                "Photo": self.file_name,
                "Date": date,
                "Attendance": attendance_status
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            logger.debug(f"DataFrame after adding new row: {df.to_dict()}")
            
            # Save to Excel
            df.to_excel(DATA_FILE, index=False)
            logger.info(f"Data saved to {DATA_FILE}")
            self.show_popup("Success", f"QR Code generated for {name} with status {attendance_status}")
            
            # Clear the input fields after submission
            self.ids.name.text = ""
            self.ids.phone.text = ""
            self.ids.id_number.text = ""
            self.ids.day.text = ""
            self.ids.month.text = ""
            self.ids.year.text = ""
            
        except Exception as e:
            logger.error(f"Error saving data to Excel: {e}")
            self.show_popup("Error", f"Failed to save data: {str(e)}")

    def select_photo(self):
        content = FileChooserListView(path=str(Path.home()), 
                                       filters=['*.png', '*.jpg', '*.jpeg'])
        self.dialog = Popup(title="Select Photo", content=content,
                            size_hint=(0.9, 0.9))
        content.bind(selection=self.handle_selected_file)
        self.dialog.open()

    def submit_data(self):
        self.generate_qr()

    def handle_selected_file(self, instance, selection):
        if selection:
            self.selected_file = selection[0]
            try:
                self.file_name = os.path.basename(self.selected_file)
                new_path = PHOTOS_DIR / self.file_name
                copyfile(self.selected_file, str(new_path))
                self.show_popup("Success", "Photo selected!")
            except Exception as e:
                self.show_popup("Error", f"Failed to select photo: {str(e)}")
            finally:
                self.dialog.dismiss()

    def show_popup(self, title, message):
            content = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint_y=None)
            content.bind(minimum_height=content.setter('height'))
    
             # Use ScrollView to ensure all text is visible
            scroll_view = ScrollView(size_hint=(1, None), height=110)  # تقليل ارتفاع الإشعار لعدم تغطية المحتوى
    
            # Label for the message with wrapping enabled
            message_label = Label(text=message, size_hint_y=None, halign='center', valign='middle', text_size=(self.width * 0.6, None))
            message_label.bind(texture_size=message_label.setter('size'))
            content.add_widget(message_label)
    
            # Close button
            close_button = Button(text="Close", size_hint_y=None, height='48dp', size_hint_x=1)
            content.add_widget(close_button)
    
            scroll_view.add_widget(content)
    
            popup = Popup(title=title, content=scroll_view, size_hint=(0.9, None), height=200, auto_dismiss=False)
            close_button.bind(on_release=popup.dismiss)
            popup.open() 

class ScanScreen(Screen):
    def __init__(self, **kwargs):
        super(ScanScreen, self).__init__(**kwargs)
        self.cap = None
        self.qr_scanned = False  # Flag to ensure QR code is scanned only once
        self.external_camera_url = None  # Variable to store external camera URL if needed

    def on_enter(self):
        # First, try to open the local camera
        self.cap = cv2.VideoCapture(0)  # 0 is typically the index for the default local camera
        
        if not self.cap.isOpened():
            logger.error("Failed to open the local camera!")
            # If local camera is not available, prompt for external camera URL
            self.prompt_for_external_camera()
        else:
            logger.debug("Local camera opened successfully")
            Clock.schedule_interval(self.update_camera, 1.0/30.0)  # 30 fps
            self.qr_scanned = False  # Reset the flag when entering the screen

    def prompt_for_external_camera(self):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.url_input = TextInput(text='http://192.168.1.6:8080/video', multiline=False)
        content.add_widget(Label(text='Enter external camera URL:'))
        content.add_widget(self.url_input)
        
        btn_layout = BoxLayout(size_hint_y=None, height='48dp', spacing=10)
        submit_btn = Button(text='Submit', size_hint_x=None, width=100)
        submit_btn.bind(on_release=self.try_external_camera)
        cancel_btn = Button(text='Cancel', size_hint_x=None, width=100)
        cancel_btn.bind(on_release=self.cancel_external_camera)
        btn_layout.add_widget(submit_btn)
        btn_layout.add_widget(cancel_btn)
        content.add_widget(btn_layout)
        
        self.external_popup = Popup(title="External Camera URL", content=content, size_hint=(0.9, 0.3))
        self.external_popup.open()

    def try_external_camera(self, instance):
        self.external_camera_url = self.url_input.text
        self.cap = cv2.VideoCapture(self.external_camera_url)
        
        if not self.cap.isOpened():
            logger.error("Failed to open the external camera!")
            self.show_popup("Error", "Could not open the external camera with the provided URL. Please try again.")
        else:
            logger.debug("External camera opened successfully")
            Clock.schedule_interval(self.update_camera, 1.0/30.0)  # 30 fps
            self.qr_scanned = False
            self.external_popup.dismiss()

    def cancel_external_camera(self, instance):
        self.show_popup("Error", "No camera available. Please provide an external camera URL or check your local camera.")
        self.external_popup.dismiss()
        
    def update_camera(self, dt):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buf = cv2.flip(frame_rgb, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                self.ids.camera_feed.texture = texture
                
                pil_image = PILImage.fromarray(frame_rgb)
                try:
                    decoded_objects = decode(pil_image)
                    if decoded_objects and not self.qr_scanned:
                        self.qr_scanned = True  # Set the flag to True to prevent multiple scans
                        self.process_qr_code(decoded_objects[0].data.decode('utf-8'))
                        # Draw a rectangle around the QR code
                        for obj in decoded_objects:
                            rect = obj.rect
                            # Convert pyzbar rect to OpenCV rect format (x, y, w, h)
                            cv2_rect = (rect.left, rect.top, rect.width, rect.height)
                            # Draw the rectangle on the frame
                            cv2.rectangle(frame_rgb, (cv2_rect[0], cv2_rect[1]), 
                                          (cv2_rect[0] + cv2_rect[2], cv2_rect[1] + cv2_rect[3]), 
                                          (0, 255, 0), 2)  # Green rectangle
                            # Update texture with the rectangle drawn
                            buf = cv2.flip(frame_rgb, 0).tobytes()
                            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
                            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                            self.ids.camera_feed.texture = texture
                except Exception as e:
                    logger.warning(f"Failed to decode QR code: {e}")
            else:
                logger.warning("Failed to capture frame")


    def process_qr_code(self, qr_data):
        try:
            _, id_number = qr_data.split(',')
            
            if DATA_FILE.exists():
                df = pd.read_excel(DATA_FILE)
                df['ID Number'] = df['ID Number'].astype(str)  # Ensure ID Number is string for comparison
                mask = df['ID Number'] == id_number
                if any(mask):
                    current_date = datetime.now().strftime("%d-%m-%Y")
                    df.loc[mask, 'Attendance'] = 'Attended'
                    df.to_excel(DATA_FILE, index=False)
                    self.show_popup("Success", f"Attendance marked for ID: {id_number}")
                else:
                    self.show_popup("Information", f"ID {id_number} not found in records. Please check registration.")
            else:
                self.show_popup("Error", "No data file available!")

        except Exception as e:
            logger.error(f"Error processing QR code: {e}")
            self.show_popup("Error", "Error processing QR code")

    def go_back(self):
        self.manager.current = 'main'

    def on_leave(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera has been released")
            Clock.unschedule(self.update_camera)

    def show_popup(self, title, message):
            content = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint_y=None)
            content.bind(minimum_height=content.setter('height'))
    
             # Use ScrollView to ensure all text is visible
            scroll_view = ScrollView(size_hint=(1, None), height=110)  # تقليل ارتفاع الإشعار لعدم تغطية المحتوى
    
            # Label for the message with wrapping enabled
            message_label = Label(text=message, size_hint_y=None, halign='center', valign='middle', text_size=(self.width * 0.6, None))
            message_label.bind(texture_size=message_label.setter('size'))
            content.add_widget(message_label)
    
            # Close button
            close_button = Button(text="Close", size_hint_y=None, height='48dp', size_hint_x=1)
            content.add_widget(close_button)
    
            scroll_view.add_widget(content)
    
            popup = Popup(title=title, content=scroll_view, size_hint=(0.9, None), height=200, auto_dismiss=False)
            close_button.bind(on_release=popup.dismiss)
            popup.open() 

class ViewScreen(Screen):
    def on_pre_enter(self):
        self.update_list()
        self.ids.filter_combo.text = "ALL"  # Default to show all

    def filter_data(self, text):
        try:
            if DATA_FILE.exists():
                df = pd.read_excel(DATA_FILE)
                # Convert all relevant columns to string to avoid AttributeError
                df['Name'] = df['Name'].astype(str)
                df['Phone'] = df['Phone'].astype(str)
                df['ID Number'] = df['ID Number'].astype(str)
                
                if text:
                    # Convert text to lowercase for case-insensitive search
                    text_lower = text.lower()
                    
                    # Filter by first and second letter of the name, case-insensitive
                    name_filter = df['Name'].str.lower().str.startswith(text_lower[:2], na=False) | df['Name'].str.lower().str.startswith(text_lower[0], na=False)
                    
                    # Filter by phone number
                    phone_filter = df['Phone'].str.contains(text, case=False, na=False)
                    
                    # Filter by ID number
                    id_filter = df['ID Number'].str.contains(text, case=False, na=False)
                    
                    # Combine filters
                    filtered_df = df[name_filter | phone_filter | id_filter]
                    self.ids.attendance_list.clear_widgets()
                    
                    for _, row in filtered_df.iterrows():
                        item_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height="120dp", spacing=15, padding=5)

                        photo_name = str(row['Photo']) if not pd.isna(row['Photo']) else "No Photo"
                        img_path = PHOTOS_DIR / photo_name
                        
                        if os.path.exists(str(img_path)):
                            img = Image(source=str(img_path), size_hint=(None, None), size=(80, 80))
                            item_layout.add_widget(img)
                        else:
                            item_layout.add_widget(Label(text="No Photo", size_hint=(None, None), size=(80, 80), halign='center', valign='middle'))

                        data_layout = BoxLayout(orientation='vertical')
                        
                        fields = ['Name', 'Phone', 'ID Number', 'Date', 'Attendance']
                        for field in fields:
                            value = str(row[field])
                            if field == 'Attendance':
                                if value == 'Invited':
                                    color = (0.5, 0.5, 0.5)  # Light Gray
                                elif value == 'Old Date' or value == 'Absent':
                                    color = (1, 0, 0)  # Red
                                elif value == 'Attended':
                                    color = (0, 1, 0)  # Green
                                else:
                                    color = (0, 0, 0)  # Black for any other status
                            else:
                                color = (0, 0, 0)  # Black for other fields

                            data_layout.add_widget(Label(
                                text=f"{field}: {value}",
                                font_size='14sp',
                                bold=True,
                                color=color
                            ))

                        item_layout.add_widget(data_layout)
                        self.ids.attendance_list.add_widget(item_layout)
                        self.ids.attendance_list.add_widget(BoxLayout(size_hint_y=None, height="10dp"))

                else:  # If no text, show all items with original colors
                    self.update_list()

        except Exception as e:
            self.show_popup("Error", f"Failed to filter data: {str(e)}")

    def update_card(self, instance, value):
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(1.0, 1.0, 1.0)  # Background color white
            Rectangle(pos=instance.pos, size=instance.size)

    def on_card_click(self, instance, index):
        if instance.active:
            self.show_delete_confirmation(index)
        else:
            # You can handle unchecking here if needed
            pass

    def update_list(self, filter_by=None):
        try:
            self.ids.attendance_list.clear_widgets()
            if DATA_FILE.exists():
                df = pd.read_excel(DATA_FILE)
                # Convert all relevant columns to string to avoid AttributeError
                df['Name'] = df['Name'].astype(str)
                df['Phone'] = df['Phone'].astype(str)
                df['ID Number'] = df['ID Number'].astype(str)
                df['Date'] = df['Date'].astype(str)
                df['Attendance'] = df['Attendance'].astype(str)
                
                # Apply filter if provided
                if filter_by and filter_by != 'ALL':
                    df = df[df['Attendance'] == filter_by]

                for index, row in df.iterrows():
                    card = BoxLayout(orientation='vertical', size_hint_y=None,
                                    height="250dp", spacing=5, padding=[10, 15, 10, 10])
                    card.bind(pos=self.update_card, size=self.update_card)

                    # Create the content of the card
                    image_layout = BoxLayout(orientation='horizontal',
                                            size_hint_y=None, height="100dp")
                    photo_name = str(row['Photo']) if not pd.isna(row['Photo']) else "No Photo"
                    img_path = PHOTOS_DIR / photo_name

                    if os.path.exists(str(img_path)):
                        img = Image(source=str(img_path), size_hint=(None, None),
                                    size=(80, 80), allow_stretch=False, keep_ratio=True)
                        image_layout.add_widget(Widget())  # Space on left
                        image_layout.add_widget(img)
                        image_layout.add_widget(Widget())  # Space on right
                    else:
                        image_layout.add_widget(Label(text="No Photo",
                                                    size_hint=(None, None),
                                                    size=(80, 80),
                                                    halign='center',
                                                    valign='middle'))

                    card.add_widget(image_layout)

                    data_layout = BoxLayout(orientation='vertical',
                                            size_hint_y=None, height="90dp")
                    text_size = '14sp'

                    fields = ['Name', 'Phone', 'ID Number', 'Date', 'Attendance']
                    for field in fields:
                        value = str(row[field])
                        color = self.get_color_based_on_status(field, value)
                        data_layout.add_widget(Label(
                            text=f"{value}",
                            font_size=text_size,
                            bold=True,
                            color=color
                        ))

                    card.add_widget(data_layout)

                    # Add trash icon at the bottom left of the card
                    trash_layout = BoxLayout(size_hint_y=None, height="40dp", padding=[5, 5, 0, 5])
                    trash_icon = ImageButton(source='Style/recycle.png', size_hint=(None, None), size=(32, 32))
                    trash_icon.bind(on_release=lambda instance, idx=index: self.show_delete_confirmation(idx))
                    trash_layout.add_widget(trash_icon)
                    trash_layout.add_widget(Widget())  # Push icon to left
                    card.add_widget(trash_layout)

                    self.ids.attendance_list.add_widget(card)
                    self.ids.attendance_list.add_widget(BoxLayout(size_hint_y=None, height="20dp"))

        except Exception as e:
            self.show_popup("Error", f"Failed to update list: {str(e)}")

    def show_delete_confirmation(self, index):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text='Do you want to delete this person?'))
        buttons = BoxLayout(size_hint_y=None, height='48dp', spacing=10)
        yes_button = MDRectangleFlatButton(text='YES', on_release=lambda x: self.delete_person(index))
        no_button = MDRectangleFlatButton(text='NO', on_release=self.cancel_delete)
        buttons.add_widget(yes_button)
        buttons.add_widget(no_button)
        content.add_widget(buttons)
        self.confirmation_popup = Popup(title="Confirm deletion", content=content, size_hint=(0.7, 0.3))
        self.confirmation_popup.open()

    def delete_person(self, index):
        try:
            # Load data from Excel file
            df = pd.read_excel(DATA_FILE)

            # Check if the index is valid to prevent errors
            if index >= len(df):
                self.show_popup("Error", "Invalid index!")
                return

            # Delete the person from the database (Excel file)
            df.drop(index, inplace=True)

            # Reset indices after deletion
            df.reset_index(drop=True, inplace=True)

            # Save changes to `data.xlsx`
            df.to_excel(DATA_FILE, index=False)

            # Update the list after deletion to remove the item from the app as well
            self.update_list(self.ids.filter_combo.text)

            # Show success message
            self.show_popup("Deleted", "Person deleted successfully!")
        
            # Close confirmation popup
            self.confirmation_popup.dismiss()

        except Exception as e:
            self.show_popup("Error", f"Failed to delete person: {str(e)}")
            self.confirmation_popup.dismiss()

    def show_popup(self, title, message):
            content = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint_y=None)
            content.bind(minimum_height=content.setter('height'))
    
             # Use ScrollView to ensure all text is visible
            scroll_view = ScrollView(size_hint=(1, None), height=110)  # تقليل ارتفاع الإشعار لعدم تغطية المحتوى
    
            # Label for the message with wrapping enabled
            message_label = Label(text=message, size_hint_y=None, halign='center', valign='middle', text_size=(self.width * 0.6, None))
            message_label.bind(texture_size=message_label.setter('size'))
            content.add_widget(message_label)
    
            # Close button
            close_button = Button(text="Close", size_hint_y=None, height='48dp', size_hint_x=1)
            content.add_widget(close_button)
    
            scroll_view.add_widget(content)
    
            popup = Popup(title=title, content=scroll_view, size_hint=(0.9, None), height=200, auto_dismiss=False)
            close_button.bind(on_release=popup.dismiss)
            popup.open() 

    def cancel_delete(self, instance):
        self.confirmation_popup.dismiss()

    def get_color_based_on_status(self, field, value):
        if field == 'Attendance':
            if value == 'Invited':
                return (0.5, 0.5, 0.5)  # Light Gray
            elif value == 'Old Date' or value == 'Absent':
                return (1, 0, 0)  # Red
            elif value == 'Attended':
                return (0, 1, 0)  # Green
            else:
                return (0, 0, 0)  # Black for any other status
        return (0, 0, 0)  # Black for other fields

    def show_qr_popup(self, index):
        df = pd.read_excel(DATA_FILE)
        name = df.iloc[index]['Name']
        phone = df.iloc[index]['Phone']
        qr_code_path = str(QR_CODES_DIR / f"{name}_{phone}.png")
        if os.path.exists(qr_code_path):
            qr_popup = Popup(title="QR Code", content=Image(source=qr_code_path, size_hint=(1, 1)), size_hint=(0.8, 0.8))
            qr_popup.open()
        else:
            self.show_popup("Error", f"QR Code for {name} not found")

    def open_filter_menu(self):
        menu_items = [
            {
                "viewclass": "OneLineListItem",
                "text": f"{i}",
                "on_release": lambda x=f"{i}": self.menu_callback(x),
            } for i in ['ALL', 'Attended', 'Invited', 'Absent']
        ]
        self.menu = MDDropdownMenu(
            caller=self.ids.filter_combo,
            items=menu_items,
            width_mult=4,
        )
        self.menu.open()

    def menu_callback(self, text_item):
        self.ids.filter_combo.text = text_item
        self.menu.dismiss()
        self.update_list(text_item)  # Assuming you want to filter based on the selection

class AttendanceApp(MDApp):
    def build(self):
        Builder.load_file('Style/kvfile.kv')
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(RegisterScreen(name='register'))
        sm.add_widget(ScanScreen(name='scan'))
        sm.add_widget(ViewScreen(name='view'))
        
        return sm

if __name__ == '__main__':
    Window.size = (360, 640)
    AttendanceApp().run()