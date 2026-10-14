import copy
import json
from pathlib import Path
import tempfile
import unittest

from merge_phase3_features import assemble, digest, BACKBONE_KEYS


class FeatureMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / 'old'
        self.new = self.root / 'new'
        self.old.mkdir()
        self.new.mkdir()
        common = [dict(seq_id=1, sequence_letter='A', residue_key=['residue'], positive=True)]
        self.rows = []
        for split, index in [('train',1),('val',2),('train',3)]:
            self.rows.append(dict(split=split, lmdb_index=index, id=str(index), cluster_pair=[str(index)],
                sides=[dict(seqres_key=str(index)+'_'+str(i), sequence_length=3,
                            sequence_sha256='sequence', common_valid=copy.deepcopy(common)) for i in (0,1)]))
        self.write(self.old/'pilot_manifest.json', self.rows[:2])
        self.write(self.old/'report.json', dict(status='protocol_ready'))
        self.write(self.new/'pilot_manifest.json', self.rows)
        self.write(self.new/'report.json', dict(status='protocol_ready', failures=[], test_records_read=0,
                                              test_labels_read=0, records_ready=3))
        self.backbone = {k:k for k in BACKBONE_KEYS}
        self.reuse = dict(status='passed', failures=[], test_records_read=0, job_id='old',
            protocol='old/report.json', protocol_sha256=digest(self.old/'report.json'),
            records=[self.feature(r) for r in self.rows[:2]], **self.backbone)
        self.shard = dict(status='passed', failures=[], test_records_read=0, job_id='new',
            protocol_sha256=digest(self.new/'report.json'), labels_outside_encoder_inputs=True,
            source_sha256='source', shard_count=1, shard_index=0, records_requested=1,
            records=[self.feature(self.rows[2])], **self.backbone)
        self.write(self.root/'reuse.json', self.reuse)
        self.write(self.root/'shard.json', self.shard)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    @staticmethod
    def feature(row):
        return dict(split=row['split'], lmdb_index=row['lmdb_index'], id=row['id'], sides=[
            dict(side=i, seqres_key=s['seqres_key'], sequence_length=3, sequence_sha256='sequence',
                 common_valid=1, gearnet_seen=1, surface_points=3,
                 cache='runs/features/'+row['id']+'_'+str(i)+'.npz') for i,s in enumerate(row['sides'])])

    def merge(self):
        return assemble(self.root,self.new/'report.json',self.root/'reuse.json',[self.root/'shard.json'])

    def test_union_preserves_old_cache_paths(self):
        report = self.merge()
        self.assertEqual(report['records_requested'],3)
        self.assertEqual(report['reused_records'],2)
        self.assertFalse(report['cache_arrays_verified'])

    def test_modified_old_label_is_rejected(self):
        self.rows[0]['sides'][0]['common_valid'][0]['positive'] = False
        self.write(self.new/'pilot_manifest.json', self.rows)
        with self.assertRaisesRegex(ValueError,'identities/labels'):
            self.merge()

    def test_missing_new_record_is_rejected(self):
        self.shard.update(records=[],records_requested=0)
        self.write(self.root/'shard.json',self.shard)
        with self.assertRaisesRegex(ValueError,'Missing feature'):
            self.merge()

    def test_different_checkpoint_is_rejected(self):
        self.shard['esm_sha256'] = 'changed'
        self.write(self.root/'shard.json',self.shard)
        with self.assertRaisesRegex(ValueError,'Backbone checkpoint'):
            self.merge()


if __name__ == '__main__':
    unittest.main()
