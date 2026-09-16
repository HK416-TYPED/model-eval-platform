import io,json,tarfile,tempfile,unittest
from pathlib import Path
from PIL import Image
from eval_platform.data_library import import_dataset


class GenericImport(unittest.TestCase):
    def test_prompt_only_jsonl_needs_no_image_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'text.jsonl'
            manifest.write_text('{"id":"one","prompt":"a mountain"}\n',encoding='utf-8')
            result=import_dataset(root,{'dataset_id':'text','source':'manifest','format':'manifest','manifest_path':str(manifest),'task':'t2i','input_fields':[],'target_field':'','count':1})
            self.assertEqual(result['count'],1)
            self.assertEqual(result['image_files'],0)

    def test_task_shapes_with_custom_fields_and_optional_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for task,count in [('edit_dual',2),('edit_single',1),('t2i',0)]:
                for target in (False,True):
                    archive=root/(task+str(target)+'.tar')
                    row={'id':'sample','instruction':'A product photo on a blue background',**{f'image{i}':f'photos/{i}.png' for i in range(count)}}
                    if target:row['expected']='photos/target.png'
                    with tarfile.open(archive,'w') as tf:
                        def add(name,data):
                            member=tarfile.TarInfo(name);member.size=len(data);tf.addfile(member,io.BytesIO(data))
                        add('metadata.jsonl',(json.dumps(row)+'\n').encode())
                        for name in [row[f'image{i}'] for i in range(count)]+([row['expected']] if target else []):
                            image=io.BytesIO();Image.new('RGB',(24,32),'green').save(image,'PNG');add(name,image.getvalue())
                    spec={'dataset_id':task+str(target),'source':'local_tar','format':'tar','archive_path':str(archive),'task':task,'input_fields':[f'image{i}' for i in range(count)],'prompt_field':'instruction','target_field':'expected','count':1}
                    record=import_dataset(root,spec)
                    data=json.loads((root/'datasets'/spec['dataset_id']/'records.jsonl').read_text(encoding='utf-8'))
                    self.assertEqual(record['task'],task)
                    self.assertEqual(len(data['inputs']),count)
                    self.assertEqual(bool(data['target']),target)
                    self.assertEqual(data['prompt'],row['instruction'])

    def test_per_sample_json_and_external_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'text.tar'
            with tarfile.open(archive,'w') as tf:
                data=json.dumps({'id':'one','prompt':'a mountain'}).encode();member=tarfile.TarInfo('one.json');member.size=len(data);tf.addfile(member,io.BytesIO(data))
            spec={'dataset_id':'text','source':'local_tar','format':'tar','archive_path':str(archive),'task':'t2i','input_fields':[],'count':1}
            self.assertEqual(import_dataset(root,spec)['count'],1)
            manifest=root/'external.jsonl';manifest.write_text('{"id":"two","prompt":"an ocean"}\n',encoding='utf-8')
            spec.update(dataset_id='external',manifest_path=str(manifest))
            self.assertEqual(import_dataset(root,spec)['selected_ids'],['two'])

    def test_missing_required_input_never_becomes_text_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'bad.tar'
            with tarfile.open(archive,'w') as tf:
                data=b'{"id":"one","prompt":"a mountain","file_name":"../escape.png"}'
                member=tarfile.TarInfo('one.json');member.size=len(data);tf.addfile(member,io.BytesIO(data))
            with self.assertRaises(ValueError):
                import_dataset(root,{'dataset_id':'bad','source':'local_tar','format':'tar','archive_path':str(archive),'task':'edit_single','input_fields':['file_name'],'count':1})
            self.assertFalse((root/'datasets/bad').exists())
