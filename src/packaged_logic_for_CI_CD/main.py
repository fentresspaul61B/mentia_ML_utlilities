import pandas as pd  # Used to store file paths for videos, audio, and images.
import os  # Used to create, iterate over, and delete files.
import re  # Used to extract start, stop, and labels from .txt file.
import subprocess  # Used to convert video to audio using ffmpeg command.
import multiprocessing  # Used for multiprocessing which optimizes efficiency.
from moviepy.editor import VideoFileClip  # Used for extracting images from video.
from PIL import Image  # Used for saving images.
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip  # Used to slice videos.
from typing import Tuple, List, Set, Any

TEST_TEXT_FILE = "tests/test_files/cat_dogs_subtitles.txt"
TEST_VIDEO = "tests/test_files/test_video_cats_and_dogs.mp4"


class JublerDataProcessor:
    def __init__(self, num_processes=2):
        self.num_processes = num_processes  # Defined for multi processing.
        pass

    @staticmethod
    def load(text_file: str) -> pd.DataFrame:
        """
        Loads in Jubler .txt file and creates a pandas dataframe from it.
        Args:
            text_file: The .txt file path to be transformed into
            pandas data frame.
        Returns:
            df: the created dataframe with start, stop, and label as columns.
        """

        # Regex used to extract the start and stop times from string in .txt file.
        time_regex = r"(\b[0-9]+\b)"

        # Regex used to extract the label from string in .txt file.
        label_regex = r"[a-zA-Z]+"
        if not os.path.exists(text_file):
            return pd.DataFrame()

        # Load the .txt data into pandas dataframe.
        df = pd.read_csv(text_file, header=None)

        # Define simple data cleaning functions that are applied to jubler df.
        def extract_time(input_str: str) -> Tuple[Any, ...]:
            return tuple(re.findall(time_regex, input_str))

        def convert_string_to_int(input_str: str) -> float:
            return int(input_str)

        def extract_label(input_str: str) -> List[Any]:
            return re.findall(label_regex, input_str)

        def clean_labels(input_str: str) -> str:
            return input_str[0] if len(input_str) else "skip"

        # Extracting time using regex.
        df["start"], df["stop"] = zip(*df[0].apply(extract_time))

        # Converting start string to int and dividing by 10 to convert MPL2 to seconds.
        df["start"] = df["start"].apply(convert_string_to_int) / 10

        # Converting stop string to int and dividing by 10 to convert MPL2 to seconds.
        df["stop"] = df["stop"].apply(convert_string_to_int) / 10

        # Extracting labels using regex.
        df["label"] = df[0].apply(extract_label)

        # Replacing empty instances with "skip" label.
        df["label"] = df["label"].apply(clean_labels)

        # Drop unneeded column.
        df = df.drop([0], axis=1)

        # Return jubler df.
        return df

    @staticmethod
    def create_nested_folders(parent_path: str, labels: Set[str]) -> str:
        """
        Creates file structure for ML training:
        parent:
            label_1:
                file_with_label_1
                file_with_label_1
                ...
            label_2:
                file_with_label_2
                file_with_label_2
                ...
            ...
            label_n:
                file_with_label_n
                ...
                last_file
        Args:
            parent_path: The path to start the nested directory structure.
            labels: The labels which will be the sub folders.
        Returns:
            parent_path
        """

        # Checking if the parent path exists, and that string is not empty.
        if not os.path.exists(parent_path) and parent_path:
            # Making the directory.
            os.mkdir(parent_path)
        try:
            # Iterating over labels to create subdirectories.
            for label in labels:
                os.mkdir(parent_path + "/" + label)
            return parent_path
        except Exception as e:
            print(e)
            return parent_path

    @staticmethod
    def slice_video(start: float,
                    stop: float,
                    label: str,
                    destination_path: str,
                    video_file: str) -> str:
        """
        Takes in a video file, and slices a smaller video from it, then
        saves that sliced video to the desired folder_path.
        Args:
            start: When the slices should begin.
            stop: When the slice ends.
            label: Label that corresponds to that video slice. Used for ML model
                training.
            destination_path: Destination for the video slice to be saved to.
            video_file: Path to the video file to be sliced.
        Returns:
            slice_video_path: The path to the new sliced video. This will be
            added to the jubler data frame for easy editing.
        """
        # Create video and sliced video path names.
        video_name = str(start) + "_" + str(stop) + "_" + label + ".mp4"
        slice_video_path = destination_path + "/" + label + "/" + video_name

        # Slice the video and save it to desired path.
        ffmpeg_extract_subclip(video_file,
                               start,
                               stop,
                               targetname=slice_video_path)

        # Return the path to the new sliced video.
        print(video_name + " Successfully sliced and saved to: " + destination_path)
        return slice_video_path

    def slice_entire_video(self,
                           destination_path: str,
                           jubler_df: pd.DataFrame,
                           video_file: str) -> pd.DataFrame:
        """
        Takes in the jubler dataframe created from load method, then slices
        the video_file into slices based on the "start" and "stop" columns from
        the jubler dataframe. Saves the sliced videos to folder_path, then adds the
        paths back into the respective rows in the jubler_data_frame.
        Args:
            jubler_df: Pandas Dataframe containing the columns:
            destination_path: Where the sliced videos will be saved.
            video_file: the input video to be sliced.
        Returns:
            jubler_data_frame: The same dataframe that is an input, but now has an
            additional column "video_path" that has the sliced videos' path.
        """

        # Create a list of tuples containing the arguments to pass to the slice_video method
        args_list = [(row.start, row.stop, row.label, destination_path, video_file) for _, row in jubler_df.iterrows()]

        # Create a pool of worker processes
        with multiprocessing.Pool(self.num_processes) as p:
            # Apply the slice_video method to each element in the args_list
            video_paths = (p.starmap(self.slice_video, args_list))

        # Add the video file paths as a column to jubler df.
        jubler_df["video_path"] = video_paths

        # Return the edited jubler df.
        return jubler_df

    @staticmethod
    def convert_video_to_audio(video_path: str,
                               start: float,
                               stop: float,
                               label: str,
                               destination_path: str,
                               output_ext: str = "wav") -> str:
        """
        Converts a single video into an audio file using ffmpeg command.
        Args:
            video_path: Path to the video to be sliced.
            start: Time in seconds when video starts in the original video.
            stop: Time in seconds when video ends in the original video.
            label: label/subtitle corresponding to video slice.
            destination_path: directory path to store data.
            output_ext: the data type for the audio. By default, it is .wav.
        Returns:
            audio_path: the path to the new audio file.
        """

        # Creating the file name for the audio file.
        file_name = f"{start}_{stop}_{label}.{output_ext}"

        # Creating the file path for the audio file.
        audio_path = f"{destination_path}/{label}/{file_name}"

        # Calling the ffmpeg command using subprocess to extract audio from a video.
        subprocess.call(["ffmpeg", "-y", "-i", video_path, audio_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)

        # Return the path to the new audio file.
        return audio_path

    @staticmethod
    def convert_video_to_images(video_path: str, start: float, stop: float, label: str, destination_path: str):
        """
        Extracts 1 image/ per second from a video clip.
        Args:
            video_path: Path to the video to be sliced.
            start: Time in seconds when video starts in the original video.
            stop: Time in seconds when video ends in the original video.
            label: label/subtitle corresponding to video data.
            destination_path: directory path to store images.
        Returns:
            images: a list of the image paths generated from the input video.
        """

        # Attempt to slice images from a video, returns None if file is corrupted.
        try:
            clip = VideoFileClip(video_path)
        except Exception as e:
            print(f"Failed to open video file {video_path}: {e}")
            return None

        # Creating a list to store the image arrays.
        images = []

        for t in range(0, int(clip.duration)):

            # Creating the path to save the image.
            frame_name = f"{start}_{stop}_{label}_{t}.png"
            frame_path = f"{destination_path}/{label}/{frame_name}"

            # Attempting to extract the image at time frame t, will skip if corrupted.
            try:
                # Extract the image from second t.
                frame = clip.get_frame(t)
                im = Image.fromarray(frame)

                # Saving the image to the desired destination.
                im.save(frame_path)

                # Adding image to save list.
                images.append(frame_path)

            # Printing out warning that an image failed to be extracted from video.
            except Exception as e:
                print(f"Failed to load image file {frame_path}: {e}, skipping.")

        # Return a list of image paths.
        return images

    def sort(self,
             jubler_df: pd.DataFrame,
             audio_destination_path: str,
             image_destination_path: str) -> bool:
        """
        Converts all videos from the video_path column in the jubler dataframe, into
        audio files. Then adds those audio paths to the jubler df.
        Args:
            jubler_df: The df that is being used to store video_path, start, stop, and label.
            audio_destination_path: Folder used to store labeled audio data.
            image_destination_path: Folder used to store labeled image data.
        Returns:
            True
        """

        # Creating lists of arguments used for multiprocessing.
        audio_args_list = []
        image_args_list = []

        # Iterate over the rows of the DataFrame
        for _, row in jubler_df.iterrows():
            # Generate the arguments for the audio extraction function and append them to the list.
            audio_args = (row.video_path, row.start, row.stop, row.label, audio_destination_path)
            audio_args_list.append(audio_args)

            # Generate the arguments for the image extraction function and append them to the list.
            image_args = (row.video_path, row.start, row.stop, row.label, image_destination_path)
            image_args_list.append(image_args)

        # Create a pool of worker processes. It does not look like it but these are loops.
        with multiprocessing.Pool(self.num_processes) as p:
            # Apply the slice_video method to each element in the args_list
            p.starmap(self.convert_video_to_audio, audio_args_list)
            p.starmap(self.convert_video_to_images, image_args_list)

        return True

    def run_pipeline(self, text_file: str, video_file: str) -> Tuple[str, str, str]:
        """
        Runs entire pipeline from start to finish. Takes Jubler txt data, and a video
        and extracts labeled audio and image data from the videos, then sorts them
        into the correct directories based on their subtitles.
        Args:
            text_file: the path to the video to be sliced.
            video_file: time when video starts in the original video.
        Returns:
            audio_destination_path (str): path to audio directory.
            image_destination_path (str): path to image directory.
            video_destination_path (str): path to video directory.
        """

        # Process text file to extract start, stop, and label.
        jubler_df = self.load(text_file)

        # Extract unique labels.
        labels = set(jubler_df.label)

        # Create folder structure based on labels extracted from text file.
        video_destination_path = self.create_nested_folders("video_data", labels)
        audio_destination_path = self.create_nested_folders("audio_data", labels)
        image_destination_path = self.create_nested_folders("image_data", labels)

        # Slicing video into smaller videos based on start and stop times.
        jubler_df = self.slice_entire_video(destination_path=video_destination_path, jubler_df=jubler_df,
                                            video_file=video_file)

        # extract and sort data.
        self.sort(jubler_df=jubler_df,
                  audio_destination_path=audio_destination_path,
                  image_destination_path=image_destination_path)
        return audio_destination_path, image_destination_path, video_destination_path


def main():
    # Run the pipeline:
    # Creating a pipeline object.
    pipeline = JublerDataProcessor()
    print(os.path.exists(TEST_TEXT_FILE))

    print(pipeline.load(TEST_TEXT_FILE).shape)

    # # Defining the txt and video files to test.
    # text_file = TEST_TEXT_FILE
    # video_file = TEST_VIDEO

    # Running the pipeline.
    # audio_data, image_data, video_data = pipeline.run_pipeline(text_file=text_file, video_file=video_file)
    # print(f"Audio data saved to: {audio_data}")
    # print(f"Image data saved to: {image_data}")
    # print("Pipeline Complete.")
    pass


if __name__ == '__main__':
    main()
