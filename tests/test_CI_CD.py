import os  # Used to edit files.
import unittest  # Used to create unit tests.
import wave  # Native python package used to validate if a file is a .wav file.
import imghdr  # Native python package used to validate if a file is a .png file.
import shutil  # Used to remove non-empty folders during testing process.
import ffmpeg  # Used to verify if a .mp4 file is corrupted.
from src.packaged_logic_for_CI_CD.main import JublerDataProcessor  # Class to test.

TEST_TEXT_FILE = "data_for_tests/cat_dog_subtitles.txt"
TEST_VIDEO = "data_for_tests/test_video_cats_and_dogs.mp4"
TEST_LABELS = {"cat", "dog", "skip"}
TEST_FILE_PATH = "tests/test_files"


# How to run tests: pytest -W ignore::DeprecationWarning
# Using the -W ignore::DeprecationWarning flag because python using python 3.7 which has older package versions.


def is_mp4_file_corrupted(file_path):
    """
    Helper function that verifies if a .mp4 file is corrupted.
    Args:
        file_path: the path to the .mp4 file
    Returns:
        True if the file is corrupted, False otherwise.
    """
    # Try to open the file with ffmpeg
    try:
        ffmpeg.probe(file_path)
        return False

    # Catch the ffmpeg.Error exception that is raised if the file is corrupted
    except ffmpeg.Error:
        return True


class TestPipeLine(unittest.TestCase):
    def setUp(self):
        self.jubler_pipeline = JublerDataProcessor()


class TestLoad(TestPipeLine):
    def test_load_1(self):
        """Tests if correct data frame is created by checking shape, and if the file exists."""
        # Testing for the correct shape.
        test_df = self.jubler_pipeline.load(TEST_TEXT_FILE)
        self.assertIsNotNone(test_df)
        self.assertEqual(test_df.shape, (101, 3))

        # Testing for the correct columns.
        test_df_columns = str(test_df.columns)
        correct_columns = "Index(['start', 'stop', 'label'], dtype='object')"
        self.assertEqual(test_df_columns, correct_columns)

    def test_load_2(self):
        """Tests if returns empty df given empty string."""
        # Testing for empty string.
        test_df = self.jubler_pipeline.load("")
        self.assertIsNotNone(test_df)
        self.assertEqual(test_df.shape, (0, 0))

        test_df_columns = str(test_df.columns)
        correct_columns = "Index([], dtype='object')"
        self.assertEqual(test_df_columns, correct_columns)


class TestNestedFolder(TestPipeLine):

    def test_create_nested_folders_1(self):
        """Tests creating the parent directory for 1 class case."""
        # Testing with 1 class, if parent directory is created properly.
        test_parent_path = "test_directory"
        test_labels = ["test_label"]
        test_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        # Testing if parent directory was created.
        self.assertIsNotNone(test_path)
        self.assertTrue(os.path.exists(test_path))
        self.assertEqual(test_path, test_parent_path)
        shutil.rmtree(test_parent_path, ignore_errors=True)

    def test_create_nested_folders_2(self):
        """Tests creating the subdirectory for 1 class case."""
        # Testing with 1 class, if subdirectory was created properly.
        test_parent_path = "test_directory"
        test_labels = ["test_label"]
        test_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        # Testing if parent directory was created.
        self.assertTrue(os.path.exists(f"{test_path}/{test_labels[0]}"))
        shutil.rmtree(test_parent_path, ignore_errors=True)

    def test_create_nested_folders_3(self):
        """Tests creating the parent directory for 3 class case."""
        # Testing with 3 classes.
        test_parent_path = "test_audio_files"
        test_labels = TEST_LABELS
        test_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        # Testing if parent directory was created.
        self.assertIsNotNone(test_path)
        self.assertTrue(os.path.exists(test_path))
        self.assertEqual(test_path, test_parent_path)
        shutil.rmtree(test_parent_path, ignore_errors=True)

    def test_create_nested_folders_4(self):
        """Tests creating the subdirectories for 3 class case."""
        # Testing if subdirectories for multiple classes were created properly.
        test_parent_path = "test_audio_files"
        test_labels = TEST_LABELS
        test_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        for label in test_labels:
            self.assertTrue(os.path.exists(f"{test_path}/{label}"))
        shutil.rmtree(test_parent_path, ignore_errors=True)

    def test_create_nested_folders_5(self):
        """Testing to check that nothing is created when given an empty string."""
        # Testing empty arguments.
        test_path = self.jubler_pipeline.create_nested_folders(parent_path="", labels="")
        correct_path = ""
        self.assertEqual(test_path, correct_path)


class TestSliceVideo(TestPipeLine):
    def test_slice_video_1(self):
        """Testing to check if a video can be sliced."""
        # Testing if correct file path is created, and a video is saved there.
        test_video = TEST_VIDEO

        # Creating nested folder structure to save data.
        test_parent_path = TEST_FILE_PATH
        test_labels = TEST_LABELS
        destination_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        start = 0
        stop = 7.1
        label = "skip"
        slice_video_path = self.jubler_pipeline.slice_video(start=start,
                                                            stop=stop,
                                                            label=label,
                                                            destination_path=destination_path,
                                                            video_file=test_video)

        # Checking if the file exists.
        self.assertTrue(os.path.exists(slice_video_path))

        # Checking if the video slice returns a corrupted file.
        self.assertFalse(is_mp4_file_corrupted(slice_video_path))

        shutil.rmtree(destination_path, ignore_errors=True)

    def test_slice_video_2(self):
        """Testing to check if a different video can be sliced."""
        # Testing if correct file path is created, and a video is saved there, for the next
        # video slice.

        # jubler_df = self.jubler_pipeline.load(TEST_TEXT_FILE)
        test_video = TEST_VIDEO

        # Creating nested folder structure to save data.
        test_parent_path = TEST_FILE_PATH
        test_labels = TEST_LABELS
        destination_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        start = 7.1
        stop = 9.1
        label = "dog"
        slice_video_path = self.jubler_pipeline.slice_video(start=start,
                                                            stop=stop,
                                                            label=label,
                                                            destination_path=destination_path,
                                                            video_file=test_video)
        # Checking if the file exists.
        # self.assertTrue(os.path.exists(slice_video_path))

        # Checking if the video slice returns a corrupted file.
        self.assertFalse(is_mp4_file_corrupted(slice_video_path))

        shutil.rmtree(destination_path, ignore_errors=True)


class TestSliceEntireVideo(TestPipeLine):
    def test_slice_entire_video(self):
        """Testing to an entire video can be sliced into smaller videos based on .txt file."""
        # Testing if correct file path is created, and a video is saved there.

        # Create and load the data frame
        jubler_df = self.jubler_pipeline.load(TEST_TEXT_FILE)
        video_file = TEST_VIDEO

        # Creating nested folder structure to save data.
        test_parent_path = TEST_FILE_PATH
        test_labels = TEST_LABELS
        destination_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        jubler_df = self.jubler_pipeline.slice_entire_video(destination_path=destination_path,
                                                            jubler_df=jubler_df,
                                                            video_file=video_file)

        self.assertTrue(os.path.exists(destination_path))
        self.assertEqual(jubler_df.shape, (101, 4))

        # Testing if all the files were created and saved properly.
        for file in jubler_df.video_path:
            self.assertTrue(os.path.exists(file))
            os.remove(file)

        shutil.rmtree(destination_path, ignore_errors=True)


class TestConvertVideoToAudioFfmpeg(TestPipeLine):
    def test_convert_video_to_audio(self):
        """Testing if a single video file can be converted to an audio file."""
        # Testing if an audio file is created, and is longer than length 0.

        # Creating nested folder structure to save data.
        test_parent_path = TEST_FILE_PATH
        test_labels = TEST_LABELS
        destination_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        video_path = TEST_VIDEO
        start = 0
        stop = 7.1
        label = "skip"
        audio_file_path = self.jubler_pipeline.convert_video_to_audio(video_path=video_path,
                                                                      start=start,
                                                                      stop=stop,
                                                                      label=label,
                                                                      destination_path=destination_path)
        self.assertTrue(os.path.exists(audio_file_path))

        # Open the audio file using the wave.open() method
        with wave.open(audio_file_path) as f:
            # Get the number of frames in the audio file
            num_frames = f.getnframes()

            # Get the frame rate (number of frames per second)
            frame_rate = f.getframerate()

            # Calculate the duration of the audio file
            duration = num_frames / frame_rate

            # Check if the duration is greater than zero
            self.assertTrue(duration > 0)

        # Remove the test file.
        shutil.rmtree(destination_path, ignore_errors=True)


class TestExtractImagesFromVideo(TestPipeLine):
    def test_convert_video_to_images(self):
        """Testing if a single video file can be converted into multiple images."""
        # Testing if an audio file is created, and is longer than length 0.

        # Creating nested folder structure to save data.
        test_parent_path = TEST_FILE_PATH
        test_labels = TEST_LABELS
        destination_path = self.jubler_pipeline.create_nested_folders(parent_path=test_parent_path, labels=test_labels)

        video_file = TEST_VIDEO
        start = 0
        stop = 7.1
        label = "skip"

        # Create short video clip:
        slice_video_path = self.jubler_pipeline.slice_video(start=start,
                                                            stop=stop,
                                                            label=label,
                                                            destination_path=destination_path,
                                                            video_file=video_file)

        # Extracting list of image paths from video.
        images = self.jubler_pipeline.convert_video_to_images(slice_video_path,
                                                              start=start,
                                                              stop=stop,
                                                              label=label,
                                                              destination_path=destination_path)
        # Iterating over each image path.
        for image_path in images[0:3]:
            # Checking if the image path exists.
            self.assertTrue(os.path.exists(image_path))

            # Use the imghdr.what() function to identify the type of the file
            file_type = imghdr.what(image_path)

            # Check if the file type is 'png'
            self.assertEqual(file_type, "png")

            # Removing the image path.
            os.remove(image_path)

        shutil.rmtree(destination_path, ignore_errors=True)
