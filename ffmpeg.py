import subprocess
import random
import re
import math


def get_audio_duration(audio_file):
    """
    Get the duration of the audio file using FFmpeg.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_file],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        # Extract duration using regex
        match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        if match:
            hours, minutes, seconds = map(float, match.groups())
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
    except Exception as e:
        print(f"Error getting audio duration: {e}")
    return None


def create_video_with_custom_durations(image_files, audio_file, subtitles_file, background_video_file, output_file):
    """
    Create a video where each image has a custom display duration, looping until the MP3 finishes,
    and transitions are added randomly between images.
    """
    try:
        # Get the total duration of the audio
        audio_duration = get_audio_duration(audio_file)
        if not audio_duration:
            print("Failed to retrieve audio duration.")
            return

        # Calculate the total duration of all images
        total_image_duration = sum(img['duration'] for img in image_files)

        # If total image duration is less than audio duration, loop images
        images_to_loop = []
        while total_image_duration < audio_duration:
            random.shuffle(image_files)  # Shuffle images for randomness
            images_to_loop.extend(image_files)
            total_image_duration += sum(img['duration'] for img in image_files)

        # Add any remaining duration to match the audio length
        remaining_duration = audio_duration - sum(img['duration'] for img in images_to_loop)
        if remaining_duration > 0:
            # Add one random image for the remaining duration
            random_image = random.choice(image_files)
            images_to_loop.append({**random_image, "duration": remaining_duration})

        # Available transitions
        transitions = [
            "fade", "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright",
            "slideup", "slidedown", "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite",
            "radial", "smoothleft", "smoothright", "smoothup", "smoothdown", "circleopen", "circleclose",
            "vertopen", "vertclose", "horzopen", "horzclose", "dissolve", "pixelize", "diagtl", "diagtr",
            "diagbl", "diagbr", "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "fadegrays", "wipetl",
            "wipetr", "wipebl", "wipebr", "squeezeh", "squeezev", "zoomin"
        ]

        # Prepare FFmpeg input and filter complex commands
        input_args = []
        filter_inputs = []
        offsets = []  # To track the offsets for each transition

        for idx, img in enumerate(images_to_loop):
            input_args += ["-loop", "1", "-t", str(math.ceil(img['duration'])), "-i", img['file']]
            filter_inputs.append(f"[{idx}:v]")

        # Get the duration of the background video
        background_video_duration = get_audio_duration(background_video_file)

        # Loop the background video if its duration is less than the audio duration
        if background_video_duration < audio_duration:
            loop_count = int(audio_duration // background_video_duration) + 1
            input_args += ["-stream_loop", str(loop_count), "-i", background_video_file]
        else:
            input_args += ["-i", background_video_file]

        # Background video filter to scale and remove audio
        background_video_filter = f"[{len(images_to_loop)}:v]scale=1080:720[a]; "

        # Create scaling and padding filters for each image
        image_filters = "".join(
            f"[{i}:v]scale=680:500:force_original_aspect_ratio=decrease,"
            f"pad=680:500:-1:-1,setsar=1[s{i}]; " for i in range(len(images_to_loop))
        )

        # Concatenate images into a video sequence
        # concat_filter = "".join(f"[s{i}]" for i in range(len(images_to_loop)))
        # concat_filter += f"concat=n={len(images_to_loop)}:v=1:a=0[image_sequence]; "

        # Overlay the image sequence on the background video

        # Add subtitles to the video
        subtitles_filter = f"[v]ass={subtitles_file}[out]"

        # Add random transitions between images
        transition_filters = []
        previous_offset = 0

        for i in range(len(images_to_loop) - 1):
            transition = random.choice(transitions)
            transition_duration = 0.5  # Adjust transition duration
            offset = previous_offset + images_to_loop[i]['duration'] - transition_duration
            if i == 0:
                transition_filters.append(f"[s0][s{i+1}]xfade=transition={transition}:duration={transition_duration}:offset={offset}[f{i}]; ")
            else:
                transition_filters.append(f"[f{i-1}][s{i+1}]xfade=transition={transition}:duration={transition_duration}:offset={offset}[f{i}]; ")
            previous_offset = offset
        overlay_filter = f"[a][f{len(images_to_loop)-2}]overlay=(W-w)/2:(H-h)/2[v]; "

        # Combine all filters into the filter_complex argument
        filter_complex = (
            background_video_filter +
            image_filters +
            # concat_filter +
            "".join(transition_filters) +
            overlay_filter +
            subtitles_filter
        )

        # Final FFmpeg command
        ffmpeg_command = [
            "ffmpeg",
            *input_args,
            "-i", audio_file,
            "-filter_complex", filter_complex,
            "-map", "[out]",  # Map the final video output
            "-map", f"{len(images_to_loop) + 1}:a",  # Map the audio stream
            "-t", str(audio_duration),  # Match the video duration to the audio
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-shortest",  # End video at the shortest input
            output_file,
        ]

    #     ffmpeg_command = [
    #        'ffmpeg',
    #        '-loop', '1', '-t', '15', '-i', 'image_3.jpg',
    #        '-loop', '1', '-t', '10', '-i', 'image_2.jpg',
    #        '-loop', '1', '-t', '5', '-i', 'image_1.jpg',
    #        '-loop', '1', '-t', '10', '-i', 'image_2.jpg',
    #        '-loop', '1', '-t', '15', '-i', 'image_3.jpg',
    #        '-loop', '1', '-t', '5', '-i', 'image_1.jpg',
    #        '-stream_loop', '6', '-i', 'background_video.mp4',
    #        '-i', 'input_audio.mp3',
    #        '-filter_complex', (
    #            # Scale background video to 1080x720
    #            '[6:v]scale=1080:720[a]; '


    #            # Scale, pad, and set SAR for each input image
    #            '[0:v]scale=680:500:force_original_aspect_ratio=decrease,pad=680:500:-1:-1,setsar=1[s0]; '
    #            '[1:v]scale=680:500:force_original_aspect_ratio=decrease,pad=680:500:-1:-1,setsar=1[s1]; '
    #            '[2:v]scale=680:500:force_original_aspect_ratio=decrease,pad=680:500:-1:-1,setsar=1[s2]; '
    #            '[3:v]scale=680:500:force_original_aspect_ratio=decrease,pad=680:500:-1:-1,setsar=1[s3]; '
    #            '[4:v]scale=680:500:force_original_aspect_ratio=decrease,pad=680:500:-1:-1,setsar=1[s4]; '
    #            '[5:v]scale=680:500:force_original_aspect_ratio=decrease,pad=680:500:-1:-1,setsar=1[s5]; '


    #            # Apply transitions between images with correct offset times
    #            '[s0][s1]xfade=transition=fadewhite:duration=1:offset=14[s01]; '
    #            '[s01][s2]xfade=transition=fadeblack:duration=1:offset=23[s02]; '
    #            '[s02][s3]xfade=transition=rectcrop:duration=1:offset=27[s03]; '
    #            '[s03][s4]xfade=transition=smoothleft:duration=1:offset=36[s04]; '
    #            '[s04][s5]xfade=transition=vertclose:duration=1:offset=50[s05]; '
    #            # '[s0][s1][s2][s3][s4][s5]concat=n=6:v=1:a=0[image_sequence]; '


    #            # Ensure the output is properly chained and overlay the image sequence on the background video
    #            '[a][s05]overlay=(W-w)/2:(H-h)/2[v]; '


    #            # Apply subtitles
    #            '[v]ass=subtitles.ass[out]'
    #        ),
    #        '-map', '[out]',  # Map the final video output
    #        '-map', '7:a',  # Map the audio input
    #        '-t', '78.98',  # Set total duration
    #        '-c:v', 'libx264',  # Use H.264 codec for video
    #        '-pix_fmt', 'yuv420p',  # Set pixel format
    #        '-shortest',  # End at the shortest input
    #        'output.mp4'
    #    ]


        # Run the FFmpeg command
        subprocess.run(ffmpeg_command, check=True)
        print(f"Video created successfully: {output_file}")

    except Exception as e:
        print(f"Error creating video: {e}")


# Input files with custom durations
image_files = [
   {"file": "image_1.jpg", "duration": 3},
   {"file": "image_2.jpg", "duration": 3},
   {"file": "image_3.jpg", "duration": 3},
]

audio_file = "input_audio.mp3"
subtitles_file = "subtitles.ass"
output_file = "output.mp4"
background_video_file = "background_video.mp4"

# Run the function
create_video_with_custom_durations(image_files, audio_file, subtitles_file, background_video_file, output_file)
