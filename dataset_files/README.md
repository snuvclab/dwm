# Example Dataset Files

These files are simplified copies of the train and test split lists used to train and evaluate the models reported in the paper.

- `trumans_train.txt`: copied from `data/dataset_files/trumans_static_pose_all/train.txt`
- `taste_rob_train.txt`: copied from `data/dataset_files/taste_rob/double_train.txt`
- `trumans_test.txt`: copied from `data/dataset_files/trumans_static_pose_all/test_selected_50.txt`
- `taste_rob_test.txt`: remapped from `data/dataset_files/taste_rob/rob_test_selected_48.txt` to the current scene-based `taste_rob/<group>/<scene>/videos/<stem>.mp4` layout

`taste_rob_test.txt` currently contains `48` entries. The legacy scene-agnostic ids `76218` and `76970` are mapped to `taste_rob/DoubleHand/Office_70513_to_77014/videos/`.

The file contents stay relative to the actual `data_root`. If you want to use these files directly, pass them as absolute paths.
