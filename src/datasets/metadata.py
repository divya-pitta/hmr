#!/usr/bin/env python3

import xml.etree.ElementTree as ET


class H36M_Metadata:
    def __init__(self, metadata_file):
        self.subjects = []
        self.sequence_mappings = {}
        self.action_names = {}
        self.camera_ids = []

        tree = ET.parse(metadata_file)
        root = tree.getroot()

        for i, tr in enumerate(root.find('mapping')):
            if i == 0:
                intrm = [td.text for td in tr]
                self.subjects = [intrm[i] for i in range(2, len(intrm))]
                self.sequence_mappings = {subject: {} for subject in self.subjects}
            elif i < 33:
                intrms = [td.text for td in tr]
                action_id  = intrms[0]
                subaction_id = intrms[1]
		prefixes = [intrms[i] for i in range(2, len(intrms))]
                for subject, prefix in zip(self.subjects, prefixes):
                    self.sequence_mappings[subject][(action_id, subaction_id)] = prefix

        for i, elem in enumerate(root.find('actionnames')):
            action_id = str(i + 1)
            self.action_names[action_id] = elem.text

        self.camera_ids = [elem.text for elem in root.find('dbcameras/index2id')]

    def get_base_filename(self, subject, action, subaction, camera):
        return '{}.{}'.format(self.sequence_mappings[subject][(action, subaction)], camera)


def load_h36m_metadata():
    return H36M_Metadata('metadata.xml')


if __name__ == '__main__':
    metadata = load_h36m_metadata()
    print(metadata.subjects)
    print(metadata.sequence_mappings)
    print(metadata.action_names)
    print(metadata.camera_ids)
