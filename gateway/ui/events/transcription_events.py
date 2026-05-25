"""Event wiring for the task creation/list tab."""

from handlers import save_media, save_text
from monitor import format_status
from runtime import format_activity_log
from services.tasks import clear_finished_task_rows, create_message, refresh_task_ui


def wire_transcription_events(t, state, timer, monitor_timer):
    monitor_timer.tick(
        fn=lambda: (format_status(), format_activity_log()),
        outputs=[t["monitor_box"], t["activity_log_box"]],
        queue=False,
        show_progress="hidden",
    )
    t["refresh_btn"].click(
        fn=lambda: (format_status(), format_activity_log()),
        outputs=[t["monitor_box"], t["activity_log_box"]],
        queue=False,
        show_progress="hidden",
    )

    task_outputs = [
        t["task_action_status"],
        state["current_uuid"],
        t["task_list_html"],
    ]

    def _after_create(result):
        message, task_id = result[0], result[1]
        return create_message(message, task_id), task_id, refresh_task_ui()

    def create_audio_task(audio_path, report_mode, sum_method):
        return _after_create(save_media(audio_path, report_mode, sum_method, source_type="audio"))

    def create_video_task(video_path, report_mode, sum_method):
        return _after_create(save_media(video_path, report_mode, sum_method, source_type="video"))

    def create_text_task(text_input, text_file, report_mode, sum_method):
        return _after_create(save_text(text_input, text_file, report_mode, sum_method))

    t["audio_btn"].click(
        fn=create_audio_task,
        inputs=[t["audio_input"], t["report_mode"], t["sum_method"]],
        outputs=task_outputs,
        queue=False,
        show_progress="hidden",
    )
    t["video_btn"].click(
        fn=create_video_task,
        inputs=[t["video_input"], t["report_mode"], t["sum_method"]],
        outputs=task_outputs,
        queue=False,
        show_progress="hidden",
    )
    t["text_btn"].click(
        fn=create_text_task,
        inputs=[t["text_input"], t["text_file"], t["report_mode"], t["sum_method"]],
        outputs=task_outputs,
        queue=False,
        show_progress="hidden",
    )

    timer.tick(
        fn=refresh_task_ui,
        outputs=[t["task_list_html"]],
        queue=False,
        show_progress="hidden",
    )
    t["refresh_tasks_btn"].click(
        fn=refresh_task_ui,
        outputs=[t["task_list_html"]],
        queue=False,
        show_progress="hidden",
    )
    t["clear_finished_btn"].click(
        fn=clear_finished_task_rows,
        outputs=[t["task_list_html"], t["task_action_status"]],
        queue=False,
        show_progress="hidden",
    )
