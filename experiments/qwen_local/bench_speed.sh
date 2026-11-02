#!/bin/zsh
# Usage: bench_speed.sh model...   -> cold load + generation speed via /api/generate
for m in "$@"; do
  curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null  # unload
  for run in cold warm; do
    curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"prompt\":\"Write three sentences about databases.\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":200}}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('$m','$run','load_s=%.1f'%(d['load_duration']/1e9),'prompt_tok=%d'%d.get('prompt_eval_count',0),'gen_tok=%d'%d['eval_count'],'tok/s=%.1f'%(d['eval_count']/(d['eval_duration']/1e9)),'total_s=%.1f'%(d['total_duration']/1e9),'| reply:',d['response'][:80].replace('\n',' '))"
  done
  curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
done
