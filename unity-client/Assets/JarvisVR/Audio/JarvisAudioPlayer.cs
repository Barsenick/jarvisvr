using System;
using System.Collections.Concurrent;
using UnityEngine;
using JarvisVR.Net;
using JarvisVR.Protocol;

namespace JarvisVR.Audio
{
    [RequireComponent(typeof(AudioSource))]
    public class JarvisAudioPlayer : MonoBehaviour
    {
        [SerializeField] private JarvisConnection connection;
        
        private ConcurrentQueue<float> _sampleQueue = new ConcurrentQueue<float>();
        private AudioSource _audioSource;
        
        private int _sourceSampleRate = 22050; // Default to Piper's rate
        private float _resampleStep = 1f;
        private float _phase = 0f;
        private float _currentSample = 0f;

        private void Awake()
        {
            _audioSource = GetComponent<AudioSource>();
            _audioSource.playOnAwake = false;
            _audioSource.loop = true;
            _audioSource.spatialBlend = 0f; // Force 2D audio
        }

        private void Start()
        {
            if (connection == null)
            {
                connection = FindObjectOfType<JarvisConnection>();
            }

            if (connection != null)
            {
                connection.OnAudioChunk += HandleAudioChunk;
            }

            // THE FIX: Create a dummy silent clip so OnAudioFilterRead is always called!
            AudioClip dummyClip = AudioClip.Create("DummySilence", 1024, 1, AudioSettings.outputSampleRate, false);
            _audioSource.clip = dummyClip;
            _audioSource.Play();
        }

        private void OnDestroy()
        {
            if (connection != null)
            {
                connection.OnAudioChunk -= HandleAudioChunk;
            }
        }

        private void HandleAudioChunk(Envelope env)
        {
            if (env.Payload == null) return;

            string b64Audio = env.Payload.Value<string>("audio");
            if (string.IsNullOrEmpty(b64Audio)) return;

            try
            {
                byte[] bytes = Convert.FromBase64String(b64Audio);
                
                if (env.Payload.ContainsKey("sr"))
                {
                    _sourceSampleRate = env.Payload.Value<int>("sr");
                    // Calculate how many output samples we need per source sample
                    // E.g., 48000 / 22050 = 2.176
                    _resampleStep = AudioSettings.outputSampleRate / (float)_sourceSampleRate;
                }

                // Convert PCM16 (short) to Float (-1.0 to 1.0)
                // We enqueue once. The resampler in OnAudioFilterRead will stretch it.
                for (int i = 0; i < bytes.Length; i += 2)
                {
                    short sample = BitConverter.ToInt16(bytes, i);
                    float floatSample = sample / 32768f;
                    _sampleQueue.Enqueue(floatSample);
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[JarvisAudioPlayer] Error decoding audio chunk: {e.Message}");
            }
        }

        private void OnAudioFilterRead(float[] data, int channels)
        {
            if (_resampleStep <= 0f) _resampleStep = 1f;

            // Loop by channels, not by raw array length
            for (int i = 0; i < data.Length; i += channels)
            {
                // If our virtual playhead is at or behind the current sample, pull a new one
                if (_phase <= 0f)
                {
                    if (_sampleQueue.TryDequeue(out float newSample))
                    {
                        _currentSample = newSample;
                        _phase += _resampleStep;
                    }
                    else
                    {
                        // Queue is empty, output silence but don't advance the phase
                        _currentSample = 0f;
                    }
                }

                // Only advance the phase if we have audio to play
                if (_phase > 0f)
                {
                    _phase -= 1f;
                }

                // Write the sample to all output channels (e.g., Mono -> Stereo)
                for (int c = 0; c < channels; c++)
                {
                    data[i + c] = _currentSample;
                }
            }
        }
    }
}