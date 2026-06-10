from kedro.pipeline import Pipeline, node

from arpi.pipelines.data_preparation.nodes.c_feed_file_sync import feed_file_sync
from arpi.pipelines.data_preparation.nodes.d_route_filtering import filter_by_route_id
from arpi.pipelines.data_preparation.nodes.e_frame_creation import write_vehicle_position_timesteps

from .nodes.a_agency_filtering import filter_gtfs_agencies
from .nodes.b_date_filtering import filter_realtime_dates


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=filter_gtfs_agencies,
                inputs={
                    "input_zip_path": "params:A_agency_filtering_input",
                    "output_zip_path": "params:A_agency_filtering_output",
                    "agencies": "params:A_agency_filtering_agencies",
                    "force": "params:A_agency_filtering_force",
                },
                outputs="agency_filtered_zip_path",
                name="filter_gtfs_agencies",
            ),
            node(
                func=filter_realtime_dates,
                inputs={
                    "input_zip_path": "agency_filtered_zip_path",
                    "output_zip_path": "params:B_date_filtering_output",
                    "start_date": "params:B_date_filtering_start_date",
                    "end_date": "params:B_date_filtering_end_date",
                    "force": "params:B_date_filtering_force",
                },
                outputs="date_filtered_zip_path",
                name="filter_gtfs_rt_archive",
            ),
            node(
                func=feed_file_sync,
                inputs={
                    "input_zip_path": "date_filtered_zip_path",
                    "output_zip_path": "params:C_feed_file_sync_output",
                    "lag_seconds": "params:C_feed_file_sync_lag_seconds",
                    "force": "params:C_feed_file_sync_force",
                },
                outputs="synced_feed_files_zip",
                name="syncronize_feed_files",
            ),
            node(
                func=filter_by_route_id,
                inputs={
                    "input_zip_path": "date_filtered_zip_path",
                    "matches_zip_path": "synced_feed_files_zip",
                    "output_zip_path": "params:D_route_filtering_output",
                    "agency": "params:D_route_filtering_agency",
                    "route_ids": "params:D_route_filtering_route_ids",
                    "force": "params:D_route_filtering_force",
                },
                outputs="route_filtered_zip",
                name="reduce_gtfs_and_matched_gtfs_rt_by_route",
            ),
            node(
                func=write_vehicle_position_timesteps,
                inputs={
                    "input_zip_path": "route_filtered_zip",
                    "matches_zip_path": "synced_feed_files_zip",
                    "output_zip_path": "params:E_frame_creation_output",
                    "force": "params:E_frame_creation_force"
                },
                outputs="travel_data_zip",
                name="generate_travel_data",
            ),
        ]
    )
